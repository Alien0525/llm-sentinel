import json
import urllib.request
import os
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EC2_URL = os.environ["OLLAMA_EC2_URL"]
GUARDRAIL_ID = os.environ["GUARDRAIL_ID"]
GUARDRAIL_VERSION = os.environ.get("GUARDRAIL_VERSION", "1")
CLASSIFIER_LAMBDA_ARN = os.environ.get("CLASSIFIER_LAMBDA_ARN")

bedrock = boto3.client("bedrock-runtime", region_name="us-east-2")
lambda_client = boto3.client("lambda", region_name="us-east-2")
cloudwatch = boto3.client("cloudwatch", region_name="us-east-2")


# ── Custom Metric ─────────────────────────────────────────────────────────────

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

        # ── Baseline mode switch ──────────────────────────────────────────────
        # Set "baseline": true in the request body to skip all guardrail layers
        # and hit TinyLlama directly. Use this for Week 2 baseline data collection.
        # Remove or set to false for Week 4 protected runs.
        baseline_mode = body.get("baseline", False)

        if not prompt:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Prompt required"})
            }

        logger.info(f"Incoming prompt: {prompt[:100]} | baseline_mode={baseline_mode}")

        if baseline_mode:
            # ── BASELINE PATH: straight to TinyLlama, no guardrails ──────────
            logger.info("BASELINE MODE: skipping all guardrail layers")
            llm_response = query_llm(prompt)
            logger.info("Baseline LLM response received")
            return respond(200, "baseline", None, llm_response)

        # ── PROTECTED PATH ────────────────────────────────────────────────────

        # Layer 1 — ML classifier
        l1 = check_layer1(prompt)
        if l1["blocked"]:
            logger.warning(f"BLOCKED by Layer1 | prompt: {prompt[:100]}")
            publish_block_metric("Layer1")
            return respond(403, "blocked", "Layer1", l1["reason"])

        # Layer 2 — Bedrock Guardrails
        l2 = check_layer2(prompt)
        if l2["blocked"]:
            logger.warning(f"BLOCKED by Layer2 | prompt: {prompt[:100]}")
            publish_block_metric("Layer2")
            return respond(403, "blocked", "Layer2", l2["reason"])

        # Passed all layers
        logger.info("Prompt passed all layers, querying LLM")
        llm_response = query_llm(prompt)
        logger.info("LLM response received successfully")
        return respond(200, "allowed", None, llm_response)

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


def respond(status, verdict, blocked_by, message):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "verdict": verdict,
            "blocked_by": blocked_by,
            "message": message
        })
    }