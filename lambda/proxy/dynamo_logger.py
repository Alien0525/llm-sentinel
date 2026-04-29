import boto3
import uuid
from datetime import datetime, timezone

dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('llm-sentinel-attack-logs')

def log_attack(raw_prompt, attack_type, layer_blocked, blocked, source_ip=None):
    item = {
        'prompt_id':     str(uuid.uuid4()),
        'timestamp':     datetime.now(timezone.utc).isoformat(),
        'attack_type':   attack_type,
        'layer_blocked': layer_blocked,
        'raw_prompt':    raw_prompt,
        'blocked':       blocked,
        'source_ip':     source_ip or 'unknown'
    }
    table.put_item(Item=item)
    return item['prompt_id']