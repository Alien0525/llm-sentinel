import boto3
import uuid
import json
from datetime import datetime, timezone

s3 = boto3.client('s3', region_name='us-east-2')
BUCKET = 'llm-sentinel-bypass-payloads'

def store_bypass_payload(prompt, attack_type='unknown'):
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    file_key = f"bypassed/{date_str}/{uuid.uuid4()}.json"
    
    payload = {
        'prompt':      prompt,
        'attack_type': attack_type,
        'timestamp':   datetime.now(timezone.utc).isoformat(),
        'label':       1
    }
    
    s3.put_object(
        Bucket=BUCKET,
        Key=file_key,
        Body=json.dumps(payload),
        ContentType='application/json'
    )
    print(f"Stored bypass payload: {file_key}")