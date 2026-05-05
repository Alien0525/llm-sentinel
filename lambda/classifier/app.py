import json
import os
import base64
import joblib


BINARY_MODEL_PATH = os.path.join(os.getcwd(), "model", "model.pkl")
INTENT_MODEL_PATH = os.path.join(os.getcwd(), "model", "intent_model.pkl")

BLOCK_THRESHOLD = 0.70

binary_model = joblib.load(BINARY_MODEL_PATH)
intent_model = joblib.load(INTENT_MODEL_PATH)


def try_decode_payload(text):
    decoded_versions = [text]

    # Try Base64 decode
    try:
        decoded = base64.b64decode(text).decode("utf-8")
        if decoded and decoded != text:
            decoded_versions.append(decoded)
    except Exception:
        pass

    # Try hex decode
    try:
        decoded = bytes.fromhex(text).decode("utf-8")
        if decoded and decoded != text:
            decoded_versions.append(decoded)
    except Exception:
        pass

    return decoded_versions


def get_probabilities(prompt):
    probabilities = binary_model.predict_proba([prompt])[0]
    classes = binary_model.classes_

    probability_map = {
        classes[i]: float(probabilities[i])
        for i in range(len(classes))
    }

    attack_confidence = probability_map.get("attack", 0.0)
    benign_confidence = probability_map.get("benign", 0.0)

    return attack_confidence, benign_confidence


def classify_single_prompt(prompt):
    label = binary_model.predict([prompt])[0]
    intent = intent_model.predict([prompt])[0]

    attack_confidence, benign_confidence = get_probabilities(prompt)

    blocked = label == "attack" and attack_confidence >= BLOCK_THRESHOLD

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
    decoded_candidates = try_decode_payload(prompt)

    best_result = None

    for candidate in decoded_candidates:
        result = classify_single_prompt(candidate)

        # If any decoded candidate is confidently malicious, block immediately
        if result["blocked"]:
            return result

        if best_result is None:
            best_result = result
        else:
            if result["attack_confidence"] > best_result["attack_confidence"]:
                best_result = result

    return best_result


def lambda_handler(event, context):
    try:
        body = event.get("body", event)

        if isinstance(body, str):
            body = json.loads(body)

        prompt = body.get("prompt", "")

        if not prompt:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                },
                "body": json.dumps({
                    "error": "Missing required field: prompt"
                })
            }

        result = classify_prompt(prompt)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps(result)
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            },
            "body": json.dumps({
                "error": str(e)
            })
        }