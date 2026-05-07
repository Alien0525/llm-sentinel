import json
import os
import base64
import uuid
from datetime import datetime, timezone

import boto3
import joblib


# =========================
# CONFIG
# =========================

MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
INTENT_MODEL_PATH = os.path.join(os.getcwd(), "intent_model.pkl")

BLOCK_THRESHOLD = float(os.environ.get("BLOCK_THRESHOLD", "0.70"))

FEEDBACK_BUCKET = os.environ.get("FEEDBACK_BUCKET", "")
FEEDBACK_PREFIX = os.environ.get("FEEDBACK_PREFIX", "feedback/")

VALID_LABELS = {"attack", "benign"}

VALID_INTENTS = {
    "prompt_injection",
    "jailbreak",
    "harmful_request",
    "pii_extraction",
    "benign"
}


# =========================
# LOAD MODELS
# =========================

binary_model = joblib.load(MODEL_PATH)

intent_model = None
if os.path.exists(INTENT_MODEL_PATH):
    intent_model = joblib.load(INTENT_MODEL_PATH)

s3_client = boto3.client("s3")


# =========================
# COMMON HELPERS
# =========================

def make_response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(body)
    }


def parse_body(event):
    body = event.get("body", event)

    if isinstance(body, str):
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    if isinstance(body, dict):
        return body

    return {}


def get_route(event, body):
    """
    Supports:
    - API Gateway HTTP API v2 rawPath
    - REST API path
    - direct Lambda test using action field
    """

    raw_path = event.get("rawPath") or event.get("path") or ""

    if raw_path:
        return raw_path

    action = body.get("action", "")

    if action == "feedback":
        return "/feedback"

    return "/classify"


def get_http_method(event):
    return (
        event.get("requestContext", {})
        .get("http", {})
        .get("method", "")
    )


# =========================
# CLASSIFICATION HELPERS
# =========================

def try_decode_payload(text):
    """
    Handles simple encoded attacks.
    Attempts Base64 and hex decoding.
    """

    decoded_versions = [text]

    # Base64
    try:
        decoded = base64.b64decode(text).decode("utf-8")
        if decoded and decoded != text:
            decoded_versions.append(decoded)
    except Exception:
        pass

    # Hex
    try:
        decoded = bytes.fromhex(text).decode("utf-8")
        if decoded and decoded != text:
            decoded_versions.append(decoded)
    except Exception:
        pass

    return decoded_versions


def get_probabilities(prompt):
    """
    Returns attack and benign probabilities from the binary classifier.
    """

    try:
        probabilities = binary_model.predict_proba([prompt])[0]
        classes = binary_model.classes_

        probability_map = {
            classes[i]: float(probabilities[i])
            for i in range(len(classes))
        }

        attack_confidence = probability_map.get("attack", 0.0)
        benign_confidence = probability_map.get("benign", 0.0)

        return attack_confidence, benign_confidence

    except Exception:
        return 0.0, 0.0


def predict_intent(prompt, label):
    """
    Uses intent_model if available.
    Falls back to simple intent mapping if intent_model.pkl is not present.
    """

    if intent_model is not None:
        try:
            return intent_model.predict([prompt])[0]
        except Exception:
            pass

    if label == "benign":
        return "benign"

    prompt_lower = prompt.lower()

    if "ignore" in prompt_lower or "system prompt" in prompt_lower or "previous instructions" in prompt_lower:
        return "prompt_injection"

    if "dan" in prompt_lower or "developer mode" in prompt_lower or "jailbreak" in prompt_lower:
        return "jailbreak"

    if "api key" in prompt_lower or "secret" in prompt_lower or "private data" in prompt_lower or "credentials" in prompt_lower:
        return "pii_extraction"

    return "harmful_request"


def classify_single_prompt(prompt):
    label = binary_model.predict([prompt])[0]

    attack_confidence, benign_confidence = get_probabilities(prompt)

    blocked = label == "attack" and attack_confidence >= BLOCK_THRESHOLD

    intent = predict_intent(prompt, label)

    return {
        "label": label,
        "intent": intent,
        "attack_confidence": round(attack_confidence, 4),
        "benign_confidence": round(benign_confidence, 4),
        "blocked": blocked,
        "threshold": BLOCK_THRESHOLD,
        "decoded_prompt_used": prompt
    }


def classify_prompt(prompt):
    """
    Classifies original and decoded versions.
    If any decoded version is confidently malicious, block immediately.
    """

    decoded_candidates = try_decode_payload(prompt)

    best_result = None

    for candidate in decoded_candidates:
        result = classify_single_prompt(candidate)

        if result["blocked"]:
            return result

        if best_result is None:
            best_result = result
        elif result["attack_confidence"] > best_result["attack_confidence"]:
            best_result = result

    return best_result


def handle_classify(body):
    prompt = body.get("prompt", "")

    if not prompt:
        return make_response(400, {
            "error": "Missing required field: prompt"
        })

    result = classify_prompt(prompt)

    return make_response(200, result)


# =========================
# FEEDBACK LOGIC
# =========================

def validate_feedback(body):
    prompt = body.get("prompt", "")
    corrected_label = str(body.get("corrected_label", "")).lower().strip()
    corrected_intent = str(body.get("corrected_intent", "")).lower().strip()
    admin_name = body.get("admin_name", "unknown")
    reason = body.get("reason", "")

    if not prompt:
        return False, "Missing required field: prompt"

    if corrected_label not in VALID_LABELS:
        return False, "corrected_label must be attack or benign"

    if corrected_intent not in VALID_INTENTS:
        return False, (
            "corrected_intent must be one of: "
            "prompt_injection, jailbreak, harmful_request, pii_extraction, benign"
        )

    if corrected_label == "benign" and corrected_intent != "benign":
        return False, "If corrected_label is benign, corrected_intent should be benign"

    if corrected_label == "attack" and corrected_intent == "benign":
        return False, "If corrected_label is attack, corrected_intent cannot be benign"

    return True, {
        "prompt": prompt,
        "corrected_label": corrected_label,
        "corrected_intent": corrected_intent,
        "admin_name": admin_name,
        "reason": reason
    }


def handle_feedback(body):
    """
    Admin supervised feedback endpoint.

    Example request:

    {
      "prompt": "Write a professional email to my professor",
      "corrected_label": "benign",
      "corrected_intent": "benign",
      "admin_name": "Adin",
      "reason": "False positive during testing"
    }
    """

    if not FEEDBACK_BUCKET:
        return make_response(500, {
            "error": "FEEDBACK_BUCKET environment variable is not set"
        })

    is_valid, validation_result = validate_feedback(body)

    if not is_valid:
        return make_response(400, {
            "error": validation_result
        })

    prompt = validation_result["prompt"]

    model_prediction = classify_prompt(prompt)

    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    feedback_record = {
        "feedback_id": feedback_id,
        "timestamp": timestamp,

        # These 3 fields are used directly during retraining
        "text": prompt,
        "label": validation_result["corrected_label"],
        "intent": validation_result["corrected_intent"],

        # Audit fields
        "admin_name": validation_result["admin_name"],
        "reason": validation_result["reason"],

        # Original model decision
        "model_label": model_prediction["label"],
        "model_intent": model_prediction["intent"],
        "model_blocked": model_prediction["blocked"],
        "model_attack_confidence": model_prediction["attack_confidence"],
        "model_benign_confidence": model_prediction["benign_confidence"],
        "threshold": model_prediction["threshold"]
    }

    safe_timestamp = timestamp.replace(":", "-")
    key = f"{FEEDBACK_PREFIX}{safe_timestamp}_{feedback_id}.json"

    s3_client.put_object(
        Bucket=FEEDBACK_BUCKET,
        Key=key,
        Body=json.dumps(feedback_record),
        ContentType="application/json"
    )

    return make_response(200, {
        "message": "Feedback saved for retraining",
        "s3_bucket": FEEDBACK_BUCKET,
        "s3_key": key,
        "feedback_record": feedback_record
    })


# =========================
# MAIN LAMBDA HANDLER
# =========================

def lambda_handler(event, context):
    try:
        body = parse_body(event)
        route = get_route(event, body)
        method = get_http_method(event)

        if method == "OPTIONS":
            return make_response(200, {
                "message": "CORS preflight success"
            })

        if route.endswith("/feedback"):
            return handle_feedback(body)

        if route.endswith("/classify"):
            return handle_classify(body)

        # Default for direct Lambda tests
        return handle_classify(body)

    except Exception as e:
        return make_response(500, {
            "error": str(e)
        })