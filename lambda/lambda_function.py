import json, urllib.request, os
from dynamo_logger import log_attack

EC2_URL = os.environ["OLLAMA_EC2_URL"]

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")
        source_ip = event.get("requestContext", {}).get("identity", {}).get("sourceIp", "unknown")
        if not prompt:
            return {"statusCode": 400, "body": json.dumps({"error": "Prompt required"})}

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
            resp = json.loads(r.read())

        log_attack(prompt, "benign", "none", False, source_ip)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"response": resp.get("response", "")})
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }