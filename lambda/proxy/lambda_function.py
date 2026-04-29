import json
import urllib.request
import os
import boto3
import logging
import base64
import time
import hashlib
import uuid
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_URL               = os.environ["OLLAMA_EC2_URL"]
GUARDRAIL_ID          = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION     = os.environ.get("GUARDRAIL_VERSION", "1")
CLASSIFIER_LAMBDA_ARN = os.environ.get("CLASSIFIER_LAMBDA_ARN")
CACHE_TTL_SECONDS     = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

bedrock        = boto3.client("bedrock-runtime", region_name="us-east-2")
lambda_client  = boto3.client("lambda",          region_name="us-east-2")
cloudwatch     = boto3.client("cloudwatch",      region_name="us-east-2")
s3             = boto3.client("s3",              region_name="us-east-2")
dynamodb       = boto3.resource("dynamodb",      region_name="us-east-2")

cache_table    = dynamodb.Table("SentinelPromptCache")
attack_table   = dynamodb.Table("llm-sentinel-attack-logs")
S3_BYPASS_BUCKET = "llm-sentinel-bypass-payloads"


# ── CloudWatch Metrics ────────────────────────────────────────────────────────

def publish_block_metric(layer):
    cloudwatch.put_metric_data(
        Namespace="LLMSentinel",
        MetricData=[{"MetricName": "BlockedRequests",
                     "Dimensions": [{"Name": "BlockedBy", "Value": layer}],
                     "Value": 1, "Unit": "Count"}]
    )

def publish_latency_metric(layer, duration_ms):
    cloudwatch.put_metric_data(
        Namespace="LLMSentinel",
        MetricData=[{"MetricName": "LayerLatency",
                     "Dimensions": [{"Name": "Layer", "Value": layer}],
                     "Value": duration_ms, "Unit": "Milliseconds"}]
    )


# ── Sisvanth: Attack Logger (DynamoDB llm-sentinel-attack-logs) ───────────────
# Called on every request — blocked or allowed — so dashboard has full data.

def log_attack(raw_prompt, attack_type, layer_blocked, blocked, source_ip=None):
    try:
        item = {
            "prompt_id":     str(uuid.uuid4()),
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "attack_type":   attack_type,
            "layer_blocked": layer_blocked or "none",
            "raw_prompt":    raw_prompt[:500],
            "blocked":       blocked,
            "source_ip":     source_ip or "unknown"
        }
        attack_table.put_item(Item=item)
        logger.info(f"Attack log written: attack_type={attack_type} blocked={blocked}")
    except Exception as e:
        logger.warning(f"log_attack failed (non-fatal): {str(e)}")


# ── Sisvanth: S3 Bypass Store ─────────────────────────────────────────────────
# Called only when a prompt passes L1+L2 but is caught by Layer 3 (egress).
# These are real bypasses — stored for Krisha's retraining pipeline.

def store_bypass_payload(prompt, attack_type="unknown"):
    try:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_key = f"bypassed/{date_str}/{uuid.uuid4()}.json"
        payload = {
            "prompt":      prompt,
            "attack_type": attack_type,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "label":       1
        }
        s3.put_object(
            Bucket=S3_BYPASS_BUCKET,
            Key=file_key,
            Body=json.dumps(payload),
            ContentType="application/json"
        )
        logger.info(f"Bypass payload stored: {file_key}")
    except Exception as e:
        logger.warning(f"store_bypass_payload failed (non-fatal): {str(e)}")


# ── DynamoDB Prompt Cache ─────────────────────────────────────────────────────

def get_prompt_hash(prompt):
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

def check_cache(prompt_hash):
    try:
        result = cache_table.get_item(Key={"prompt_hash": prompt_hash})
        item = result.get("Item")
        if item and int(time.time()) < item.get("expires_at", 0):
            logger.info(f"Cache HIT: {prompt_hash[:12]}...")
            return item
    except Exception as e:
        logger.warning(f"Cache read error (non-fatal): {str(e)}")
    return None

def write_cache(prompt_hash, verdict, blocked_by, message, encoding):
    try:
        cache_table.put_item(Item={
            "prompt_hash":       prompt_hash,
            "verdict":           verdict,
            "blocked_by":        blocked_by or "none",
            "message":           message[:500],
            "encoding_detected": encoding or "none",
            "cached_at":         int(time.time()),
            "expires_at":        int(time.time()) + CACHE_TTL_SECONDS
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
                logger.warning(f"Base64 detected. Decoded: {decoded_text[:60]}")
                return decoded_text, "base64"
        except Exception:
            pass
    hex_candidate = stripped.replace(" ", "").replace("0x", "")
    if len(hex_candidate) >= 20 and len(hex_candidate) % 2 == 0:
        try:
            decoded_bytes = bytes.fromhex(hex_candidate)
            decoded_text = decoded_bytes.decode("utf-8")
            if sum(c.isprintable() for c in decoded_text) / len(decoded_text) > 0.85:
                logger.warning(f"Hex detected. Decoded: {decoded_text[:60]}")
                return decoded_text, "hex"
        except Exception:
            pass
    return prompt, None


# ── Layer 1 — ML Classifier ───────────────────────────────────────────────────

def check_layer1(prompt):
    if not CLASSIFIER_LAMBDA_ARN:
        logger.info("Layer1: skipped (CLASSIFIER_LAMBDA_ARN not set)")
        return {"blocked": False, "reason": "Layer1 not deployed"}
    try:
        response = lambda_client.invoke(
            FunctionName=CLASSIFIER_LAMBDA_ARN,
            InvocationType="RequestResponse",
            Payload=json.dumps({"prompt": prompt}).encode()
        )
        result = json.loads(response["Payload"].read())
        logger.info(f"Layer1: blocked={result['blocked']} reason={result.get('reason','')}")
        return result
    except Exception as e:
        logger.error(f"Layer1 error: {str(e)}")
        return {"blocked": False, "reason": f"Layer1 error: {str(e)}"}


# ── Layer 2 — Bedrock Guardrails Input ───────────────────────────────────────

def check_layer2(prompt):
    response = bedrock.apply_guardrail(
        guardrailIdentifier=GUARDRAIL_ID,
        guardrailVersion=GUARDRAIL_VERSION,
        source="INPUT",
        content=[{"text": {"text": prompt}}]
    )
    blocked = response["action"] == "GUARDRAIL_INTERVENED"
    reason  = response.get("outputs", [{}])[0].get("text", "") if blocked else ""
    logger.info(f"Layer2: blocked={blocked} reason={reason}")
    return {"blocked": blocked, "reason": reason}


# ── EC2 Ollama / TinyLlama ────────────────────────────────────────────────────

def query_llm(prompt):
    payload = json.dumps({
        "model": "tinyllama", "prompt": prompt, "stream": False,
        "options": {"num_predict": 30, "temperature": 0.2}
    }).encode()
    req = urllib.request.Request(
        f"{EC2_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("response", "")


# ── Layer 3 — Bedrock Guardrails Output (Egress Auditor) ─────────────────────

def check_layer3(llm_response):
    try:
        response = bedrock.apply_guardrail(
            guardrailIdentifier=GUARDRAIL_ID,
            guardrailVersion=GUARDRAIL_VERSION,
            source="OUTPUT",
            content=[{"text": {"text": llm_response}}]
        )
        blocked = response["action"] == "GUARDRAIL_INTERVENED"
        reason  = response.get("outputs", [{}])[0].get("text", "") if blocked else ""
        logger.info(f"Layer3: blocked={blocked} reason={reason}")
        return {"blocked": blocked, "reason": reason}
    except Exception as e:
        logger.error(f"Layer3 error: {str(e)}")
        return {"blocked": False, "reason": f"Layer3 error: {str(e)}"}


# ── Handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        body          = json.loads(event.get("body", "{}"))
        prompt        = body.get("prompt", "")
        baseline_mode = body.get("baseline", False)
        source_ip     = event.get("requestContext", {}).get("http", {}).get("sourceIp", "unknown")

        if not prompt:
            return {"statusCode": 400, "body": json.dumps({"error": "Prompt required"})}

        logger.info(f"Prompt: {prompt[:100]} | baseline={baseline_mode} | ip={source_ip}")

        # ── Baseline: no layers, no cache, no logging ─────────────────────────
        if baseline_mode:
            t0 = time.time()
            llm_response = query_llm(prompt)
            publish_latency_metric("EC2_TinyLlama", (time.time() - t0) * 1000)
            return respond(200, "baseline", None, llm_response, encoding=None, cached=False)

        # ── Cache check ───────────────────────────────────────────────────────
        prompt_hash = get_prompt_hash(prompt)
        t_cache = time.time()
        cached_item = check_cache(prompt_hash)
        publish_latency_metric("CacheLookup", (time.time() - t_cache) * 1000)

        if cached_item:
            log_attack(
                raw_prompt=prompt,
                attack_type=cached_item.get("blocked_by", "none"),
                layer_blocked=cached_item.get("blocked_by") if cached_item.get("blocked_by") != "none" else None,
                blocked=(cached_item["verdict"] == "blocked"),
                source_ip=source_ip
            )
            status = 403 if cached_item["verdict"] == "blocked" else 200
            return respond(
                status, cached_item["verdict"],
                cached_item.get("blocked_by") if cached_item.get("blocked_by") != "none" else None,
                cached_item["message"],
                encoding=cached_item.get("encoding_detected") if cached_item.get("encoding_detected") != "none" else None,
                cached=True
            )

        # ── Decode ────────────────────────────────────────────────────────────
        t_decode = time.time()
        decoded_prompt, encoding_detected = decode_if_encoded(prompt)
        publish_latency_metric("Decode", (time.time() - t_decode) * 1000)

        # ── Layer 1 ───────────────────────────────────────────────────────────
        t1 = time.time()
        l1 = check_layer1(decoded_prompt)
        publish_latency_metric("Layer1_Classifier", (time.time() - t1) * 1000)

        if l1["blocked"]:
            publish_block_metric("Layer1")
            write_cache(prompt_hash, "blocked", "Layer1", l1["reason"], encoding_detected)
            log_attack(prompt, attack_type="prompt_injection", layer_blocked="Layer1",
                       blocked=True, source_ip=source_ip)
            return respond(403, "blocked", "Layer1", l1["reason"],
                           encoding=encoding_detected, cached=False)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        t2 = time.time()
        l2 = check_layer2(decoded_prompt)
        publish_latency_metric("Layer2_Bedrock", (time.time() - t2) * 1000)

        if l2["blocked"]:
            publish_block_metric("Layer2")
            write_cache(prompt_hash, "blocked", "Layer2", l2["reason"], encoding_detected)
            log_attack(prompt, attack_type="jailbreak", layer_blocked="Layer2",
                       blocked=True, source_ip=source_ip)
            return respond(403, "blocked", "Layer2", l2["reason"],
                           encoding=encoding_detected, cached=False)

        # ── EC2 ───────────────────────────────────────────────────────────────
        t3 = time.time()
        llm_response = query_llm(prompt)
        publish_latency_metric("EC2_TinyLlama", (time.time() - t3) * 1000)

        # ── Layer 3 ───────────────────────────────────────────────────────────
        t4 = time.time()
        l3 = check_layer3(llm_response)
        publish_latency_metric("Layer3_Egress", (time.time() - t4) * 1000)

        if l3["blocked"]:
            publish_block_metric("Layer3")
            store_bypass_payload(decoded_prompt, attack_type="egress_bypass")
            log_attack(prompt, attack_type="egress_bypass", layer_blocked="Layer3",
                       blocked=True, source_ip=source_ip)
            return respond(403, "blocked", "Layer3", l3["reason"],
                           encoding=encoding_detected, cached=False)

        # ── All clear ─────────────────────────────────────────────────────────
        write_cache(prompt_hash, "allowed", None, llm_response, encoding_detected)
        log_attack(prompt, attack_type="none", layer_blocked=None,
                   blocked=False, source_ip=source_ip)
        return respond(200, "allowed", None, llm_response,
                       encoding=encoding_detected, cached=False)

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def respond(status, verdict, blocked_by, message, encoding=None, cached=False):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "verdict":           verdict,
            "blocked_by":        blocked_by,
            "encoding_detected": encoding,
            "cached":            cached,
            "message":           message
        })
    }