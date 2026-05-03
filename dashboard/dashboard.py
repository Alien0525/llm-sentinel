import streamlit as st
import boto3
from collections import Counter

# ── DynamoDB connection ───────────────────────────────────────────────────────
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('llm-sentinel-attack-logs')

st.set_page_config(page_title="LLM Sentinel Dashboard", layout="wide")
st.title("🛡️ LLM Sentinel — Real-Time Attack Dashboard")

# ── Fetch all logs ────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_logs():
    result = table.scan()
    return result.get('Items', [])

items = fetch_logs()

if not items:
    st.warning("No data in DynamoDB yet.")
    st.stop()

# ── Metrics ───────────────────────────────────────────────────────────────────
total = len(items)
blocked = sum(1 for i in items if i.get('blocked') == True)
allowed = total - blocked
block_rate = (blocked / total * 100) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Requests", total)
col2.metric("Blocked", blocked)
col3.metric("Allowed", allowed)
col4.metric("Block Rate", f"{block_rate:.1f}%")

st.divider()

# ── Layer Breakdown ───────────────────────────────────────────────────────────
st.subheader("Layer Breakdown")
layer_counts = Counter(i.get('layer_blocked', 'unknown') for i in items if i.get('blocked') == True)

col1, col2, col3 = st.columns(3)
col1.metric("Layer 1 Blocks", layer_counts.get('layer_1', 0))
col2.metric("Layer 2 Blocks", layer_counts.get('layer_2', 0))
col3.metric("Layer 3 Blocks", layer_counts.get('layer_3', 0))

st.divider()

# ── Attack Type Breakdown ─────────────────────────────────────────────────────
st.subheader("Attack Types")
attack_counts = Counter(i.get('attack_type', 'unknown') for i in items)
st.bar_chart(attack_counts)

st.divider()

# ── Recent Logs ───────────────────────────────────────────────────────────────
st.subheader("Recent Logs")
sorted_items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
for item in sorted_items:
    st.write(f"`{item.get('timestamp', '')[:19]}` | **{item.get('attack_type')}** | Layer: `{item.get('layer_blocked')}` | Blocked: `{item.get('blocked')}`")
