import json
import urllib.request
import os
import boto3
import logging
import base64
import binascii
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_URL = os.environ["OLLAMA_EC2_URL"]
GUARDRAIL_ID = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")
CLASSIFIER_LAMBDA_ARN = os.environ.get("CLASSIFIER_LAMBDA_ARN")

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")
lambda_client = boto3.client("lambda", region_name="us-east-2")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")


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
    """Publish per-layer latency to CloudWatch for profiling."""
    cloudwatch.put_metric_data(
        Namespace="LLMSentinel",
        MetricData=[{
            "MetricName": "LayerLatency",
            "Dimensions": [{"Name": "Layer", "Value": layer}],
            "Value": duration_ms,
            "Unit": "Milliseconds"
        }]
    )


# ── Week 5: Encoded Payload Detection ─────────────────────────────────────────

def decode_if_encoded(prompt):
    """
    Attempt to decode Base64 or hex-encoded payloads before classification.
    Attackers encode prompts to bypass plain-text classifiers.

    Returns (decoded_prompt, encoding_detected):
      - decoded_prompt: the decoded text if encoding found, else original
      - encoding_detected: "base64" | "hex" | None
    """

    # ── Try Base64 ────────────────────────────────────────────────────────────
    # Base64 strings are typically longer and end with = padding.
    # We only decode if the result is valid UTF-8 printable text.
    stripped = prompt.strip()
    # Only attempt decode if input looks like base64 (no spaces, mostly b64 chars)
    import re as _re
    b64_char_ratio = len(_re.sub(r"[^A-Za-z0-9+/=]", "", stripped)) / max(len(stripped), 1)
    if len(stripped) > 30 and b64_char_ratio > 0.90:
        try:
            # Add padding if missing (some encoders strip it)
            padded = stripped + "=" * (-len(stripped) % 4)
            decoded_bytes = base64.b64decode(padded, validate=True)
            decoded_text = decoded_bytes.decode("utf-8")
            # Sanity check: decoded result should be mostly printable ASCII
            # AND must be longer than 10 chars (avoids short flukes)
            printable_ratio = sum(c.isprintable() for c in decoded_text) / len(decoded_text)
            if printable_ratio > 0.85 and len(decoded_text) > 10:
                logger.warning(f"Base64 encoded payload detected. Original: {stripped[:60]} | Decoded: {decoded_text[:60]}")
                return decoded_text, "base64"
        except Exception:
            pass  # Not valid base64, continue

    # ── Try Hex ───────────────────────────────────────────────────────────────
    # Hex-encoded strings contain only 0-9 a-f characters and are even-length.
    hex_candidate = stripped.replace(" ", "").replace("0x", "")
    if len(hex_candidate) >= 20 and len(hex_candidate) % 2 == 0:
        try:
            decoded_bytes = bytes.fromhex(hex_candidate)
            decoded_text = decoded_bytes.decode("utf-8")
            printable_ratio = sum(c.isprintable() for c in decoded_text) / len(decoded_text)
            if printable_ratio > 0.85 and len(decoded_text) > 5:
                logger.warning(f"Hex encoded payload detected. Original: {stripped[:60]} | Decoded: {decoded_text[:60]}")
                return decoded_text, "hex"
        except Exception:
            pass  # Not valid hex, continue

    return prompt, None


# ── Layer 1 — ML Classifier ───────────────────────────────────────────────────

def check_layer1(prompt):
    if not CLASSIFIER_LAMBDA_ARN:
        logger.info("Layer1: skipped (not deployed)")
        return {"blocked": False, "reason": "Layer1 not deployed"}
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


# ── Layer 2 — Bedrock Guardrails ──────────────────────────────────────────────

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
        "options": {
            "num_predict": 30,
            "temperature": 0.2
        }
    }).encode()
    req = urllib.request.Request(
        f"{EC2_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("response", "")


# ── Handler ───────────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")
        baseline_mode = body.get("baseline", False)

        if not prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Prompt required"})
            }

        logger.info(f"Incoming prompt: {prompt[:100]} | baseline_mode={baseline_mode}")

        # ── Baseline path: straight to TinyLlama, no guardrails ───────────────
        if baseline_mode:
            logger.info("BASELINE MODE: skipping all guardrail layers")
            t0 = time.time()
            llm_response = query_llm(prompt)
            publish_latency_metric("EC2_TinyLlama", (time.time() - t0) * 1000)
            logger.info("Baseline LLM response received")
            return respond(200, "baseline", None, llm_response, encoding=None)

        # ── Week 5: Decode encoded payloads before classification ─────────────
        t_decode = time.time()
        decoded_prompt, encoding_detected = decode_if_encoded(prompt)
        decode_ms = (time.time() - t_decode) * 1000
        publish_latency_metric("Decode", decode_ms)

        if encoding_detected:
            logger.warning(f"Encoding detected: {encoding_detected} | Running classifier on decoded prompt")

        # ── Layer 1 — ML classifier (on decoded prompt) ───────────────────────
        t1 = time.time()
        l1 = check_layer1(decoded_prompt)
        publish_latency_metric("Layer1_Classifier", (time.time() - t1) * 1000)

        if l1["blocked"]:
            logger.warning(f"BLOCKED by Layer1 | encoding={encoding_detected} | prompt: {prompt[:100]}")
            publish_block_metric("Layer1")
            return respond(403, "blocked", "Layer1", l1["reason"], encoding=encoding_detected)

        # ── Layer 2 — Bedrock Guardrails ──────────────────────────────────────
        t2 = time.time()
        l2 = check_layer2(decoded_prompt)
        publish_latency_metric("Layer2_Bedrock", (time.time() - t2) * 1000)

        if l2["blocked"]:
            logger.warning(f"BLOCKED by Layer2 | encoding={encoding_detected} | prompt: {prompt[:100]}")
            publish_block_metric("Layer2")
            return respond(403, "blocked", "Layer2", l2["reason"], encoding=encoding_detected)

        # ── EC2 / TinyLlama ───────────────────────────────────────────────────
        logger.info("Prompt passed all layers, querying LLM")
        t3 = time.time()
        llm_response = query_llm(prompt)  # send original prompt to LLM, not decoded
        publish_latency_metric("EC2_TinyLlama", (time.time() - t3) * 1000)

        logger.info("LLM response received successfully")
        return respond(200, "allowed", None, llm_response, encoding=encoding_detected)

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def respond(status, verdict, blocked_by, message, encoding=None):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "verdict": verdict,
            "blocked_by": blocked_by,
            "encoding_detected": encoding,   # NEW: tells team if encoded attack was caught
            "message": message
        })
    }