import json
import urllib.request
import os
import boto3
import logging
import base64
import time
import hashlib

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_URL = os.environ["OLLAMA_EC2_URL"]
GUARDRAIL_ID = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

# Classifier Lambda (deployed from ECR image)
CLASSIFIER_LAMBDA_ARN = os.environ.get("CLASSIFIER_LAMBDA_ARN")

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")
lambda_client = boto3.client("lambda", region_name="us-east-2")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")
dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
cache_table = dynamodb.Table("SentinelPromptCache")


# ── Custom Metrics ────────────────────────────────────────────────────────────

def publish_block_metric(layer):
    cloudwatch.put_metric_data(
        Namespace="LLMSentinel",
        MetricData=[{
            "MetricName": "BlockedRequests",
            "Dimensions": [{"Name": "BlockedBy", "Value": layer}],
            "Value": 1,
            "Unit": "Count"
        }]
    )

def publish_latency_metric(layer, duration_ms):
    cloudwatch.put_metric_data(
        Namespace="LLMSentinel",
        MetricData=[{
            "MetricName": "LayerLatency",
            "Dimensions": [{"Name": "Layer", "Value": layer}],
            "Value": duration_ms,
            "Unit": "Milliseconds"
        }]
    )


# ── DynamoDB Prompt Cache ─────────────────────────────────────────────────────

def get_prompt_hash(prompt):
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

def check_cache(prompt_hash):
    try:
        result = cache_table.get_item(Key={"prompt_hash": prompt_hash})
        item = result.get("Item")
        if item and int(time.time()) < item.get("expires_at", 0):
            logger.info(f"Cache HIT for hash {prompt_hash[:12]}...")
            return item
    except Exception as e:
        logger.warning(f"Cache read error (non-fatal): {str(e)}")
    return None

def write_cache(prompt_hash, verdict, blocked_by, message, encoding):
    try:
        cache_table.put_item(Item={
            "prompt_hash": prompt_hash,
            "verdict": verdict,
            "blocked_by": blocked_by or "none",
            "message": message[:500],
            "encoding_detected": encoding or "none",
            "cached_at": int(time.time()),
            "expires_at": int(time.time()) + CACHE_TTL_SECONDS
        })
    except Exception as e:
        logger.warning(f"Cache write error (non-fatal): {str(e)}")


# ── Encoded Payload Detection ─────────────────────────────────────────────────

def decode_if_encoded(prompt):
    stripped = prompt.strip()
    if len(stripped) > 20:
        try:
            padded = stripped + "=" * (-len(stripped) % 4)
            decoded_bytes = base64.b64decode(padded, validate=True)
            decoded_text = decoded_bytes.decode("utf-8")
            if sum(c.isprintable() for c in decoded_text) / len(decoded_text) > 0.85:
                logger.warning(f"Base64 payload detected. Decoded: {decoded_text[:60]}")
                return decoded_text, "base64"
        except Exception:
            pass

    hex_candidate = stripped.replace(" ", "").replace("0x", "")
    if len(hex_candidate) >= 20 and len(hex_candidate) % 2 == 0:
        try:
            decoded_bytes = bytes.fromhex(hex_candidate)
            decoded_text = decoded_bytes.decode("utf-8")
            if sum(c.isprintable() for c in decoded_text) / len(decoded_text) > 0.85:
                logger.warning(f"Hex payload detected. Decoded: {decoded_text[:60]}")
                return decoded_text, "hex"
        except Exception:
            pass

    return prompt, None


# ── Layer 1 — ML Classifier (Krisha's containerised TF-IDF model) ─────────────

def check_layer1(prompt):
    try:
        response = lambda_client.invoke(
            FunctionName=CLASSIFIER_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps({"prompt": prompt}).encode()
        )
        result = json.loads(response["Payload"].read())
        logger.info(f"Layer1: blocked={result['blocked']} reason={result.get('reason', '')}")
        return result
    except Exception as e:
        logger.error(f"Layer1 error: {str(e)}")
        return {"blocked": False, "reason": f"Layer1 error: {str(e)}"}


# ── Layer 2 — Bedrock Guardrails (input semantic check) ───────────────────────

def check_layer2(prompt):
    response = bedrock.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="INPUT",
        content=[{"text": {"text": prompt}}]
    )
    blocked = response["action"] == "GUARDRAIL_INTERVENED"
    reason = response.get("outputs", [{}])[0].get("text", "") if blocked else ""
    logger.info(f"Layer2: blocked={blocked} reason={reason}")
    return {"blocked": blocked, "reason": reason}


# ── EC2 Ollama ────────────────────────────────────────────────────────────────

def query_llm(prompt):
    payload = json.dumps({
        "model": "tinyllama",
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": 30, "temperature": 0.2}
    }).encode()
    req = urllib.request.Request(
        f"{EC2_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("response", "")


# ── Layer 3 — Bedrock Guardrails (output / egress audit) ─────────────────────
# Varsha's Week 4 task: applied to LLM response before returning to user.
# Prevents TinyLlama from leaking system prompt content, PII, or secrets
# even if the input guardrails were bypassed somehow.

def check_layer3(llm_response):
    try:
        response = bedrock.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="OUTPUT",                        # OUTPUT mode — audits the response
            content=[{"text": {"text": llm_response}}]
        )
        blocked = response["action"] == "GUARDRAIL_INTERVENED"
        reason = response.get("outputs", [{}])[0].get("text", "") if blocked else ""
        logger.info(f"Layer3: blocked={blocked} reason={reason}")
        return {"blocked": blocked, "reason": reason}
    except Exception as e:
        logger.error(f"Layer3 error: {str(e)}")
        # Non-fatal: if egress check fails, still return the response
        # but log it for investigation
        return {"blocked": False, "reason": f"Layer3 error: {str(e)}"}


# ── Handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")
        baseline_mode = body.get("baseline", False)

        if not prompt:
            return {"statusCode": 400, "body": json.dumps({"error": "Prompt required"})}

        logger.info(f"Incoming prompt: {prompt[:100]} | baseline_mode={baseline_mode}")

        # ── Baseline: straight to TinyLlama, no layers, no cache ─────────────
        if baseline_mode:
            logger.info("BASELINE MODE: skipping all guardrail layers and cache")
            t0 = time.time()
            llm_response = query_llm(prompt)
            publish_latency_metric("EC2_TinyLlama", (time.time() - t0) * 1000)
            return respond(200, "baseline", None, llm_response, encoding=None, cached=False)

        # ── Cache check ───────────────────────────────────────────────────────
        prompt_hash = get_prompt_hash(prompt)
        t_cache = time.time()
        cached = check_cache(prompt_hash)
        publish_latency_metric("CacheLookup", (time.time() - t_cache) * 1000)

        if cached:
            logger.info(f"Returning cached verdict: {cached['verdict']}")
            status = 403 if cached["verdict"] == "blocked" else 200
            return respond(
                status,
                cached["verdict"],
                cached.get("blocked_by") if cached.get("blocked_by") != "none" else None,
                cached["message"],
                encoding=cached.get("encoding_detected") if cached.get("encoding_detected") != "none" else None,
                cached=True
            )

        # ── Decode encoded payloads before classification ─────────────────────
        t_decode = time.time()
        decoded_prompt, encoding_detected = decode_if_encoded(prompt)
        publish_latency_metric("Decode", (time.time() - t_decode) * 1000)

        # ── Layer 1 — ML classifier ───────────────────────────────────────────
        t1 = time.time()
        l1 = check_layer1(decoded_prompt)
        publish_latency_metric("Layer1_Classifier", (time.time() - t1) * 1000)

        if l1["blocked"]:
            publish_block_metric("Layer1")
            write_cache(prompt_hash, "blocked", "Layer1", l1["reason"], encoding_detected)
            return respond(403, "blocked", "Layer1", l1["reason"], encoding=encoding_detected, cached=False)

        # ── Layer 2 — Bedrock Guardrails (input) ──────────────────────────────
        t2 = time.time()
        l2 = check_layer2(decoded_prompt)
        publish_latency_metric("Layer2_Bedrock", (time.time() - t2) * 1000)

        if l2["blocked"]:
            publish_block_metric("Layer2")
            write_cache(prompt_hash, "blocked", "Layer2", l2["reason"], encoding_detected)
            return respond(403, "blocked", "Layer2", l2["reason"], encoding=encoding_detected, cached=False)

        # ── EC2 / TinyLlama ───────────────────────────────────────────────────
        logger.info("Prompt passed input layers, querying LLM")
        t3 = time.time()
        llm_response = query_llm(prompt)
        publish_latency_metric("EC2_TinyLlama", (time.time() - t3) * 1000)

        # ── Layer 3 — Bedrock Guardrails (output / egress audit) ──────────────
        t4 = time.time()
        l3 = check_layer3(llm_response)
        publish_latency_metric("Layer3_Egress", (time.time() - t4) * 1000)

        if l3["blocked"]:
            publish_block_metric("Layer3")
            # Do NOT cache a blocked egress — the input was fine, only output was unsafe
            # Next time the same prompt should still reach TinyLlama and get a fresh response
            return respond(403, "blocked", "Layer3", l3["reason"], encoding=encoding_detected, cached=False)

        # All layers passed — cache and return
        write_cache(prompt_hash, "allowed", None, llm_response, encoding_detected)
        return respond(200, "allowed", None, llm_response, encoding=encoding_detected, cached=False)

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def respond(status, verdict, blocked_by, message, encoding=None, cached=False):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "verdict": verdict,
            "blocked_by": blocked_by,
            "encoding_detected": encoding,
            "cached": cached,
            "message": message
        })
    }