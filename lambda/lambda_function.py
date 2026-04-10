import json, urllib.request, os

EC2_URL = os.environ["OLLAMA_EC2_URL"]

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        prompt = body.get("prompt", "")

        if not prompt:
            return {"statusCode": 400, "body": json.dumps({"error": "Prompt required"})}

        if len(prompt) > 500:
            return {"statusCode": 400, "body": json.dumps({"error": "Prompt too long"})}

        payload = json.dumps({
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 30
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