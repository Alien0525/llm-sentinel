import streamlit as st
import boto3
import json
import urllib.request
from collections import Counter

# ── DynamoDB connection ───────────────────────────────────────────────────────
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('llm-sentinel-attack-logs')

st.set_page_config(page_title="LLM Sentinel Dashboard", layout="wide")
st.title("🛡️ LLM Sentinel — Real-Time Attack Dashboard")

# ── Fetch all logs from DynamoDB ──────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_logs():
    result = table.scan()
    return result.get('Items', [])

# ── Fetch benchmark data from GitHub ─────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_github_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())

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

# ── Before / After Comparison ─────────────────────────────────────────────────
st.subheader("📊 Before vs After — OWASP LLM Top 10 Benchmark")

BASELINE_URL  = "https://raw.githubusercontent.com/Alien0525/llm-sentinel/refs/heads/main/results/baseline_results.json"
PROTECTED_URL = "https://raw.githubusercontent.com/Alien0525/llm-sentinel/refs/heads/main/results/week5_benchmark_results.json"

try:
    baseline  = fetch_github_json(BASELINE_URL)
    protected_all = fetch_github_json(PROTECTED_URL)
    # Use plain text pass only for fair comparison with baseline
    protected = [i for i in protected_all if i.get('pass') == 'plain']

    # Summary metrics
    total_b = len(baseline)
    total_p = len(protected)
    baseline_blocked  = sum(1 for i in baseline  if i.get('result') == 'BLOCKED')
    baseline_complied = sum(1 for i in baseline  if i.get('result') == 'COMPLIED')
    protected_blocked  = sum(1 for i in protected if i.get('result') == 'BLOCKED')
    protected_complied = sum(1 for i in protected if i.get('result') == 'COMPLIED')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔴 Before (No Defense Layers)")
        st.metric("Total Payloads", total_b)
        st.metric("Blocked", baseline_blocked)
        st.metric("Complied", baseline_complied)
        st.metric("Block Rate", f"{baseline_blocked/total_b*100:.1f}%")

    with col2:
        st.markdown("### 🟢 After (All 3 Layers Active)")
        st.metric("Total Payloads", total_p)
        st.metric("Blocked", protected_blocked, delta=f"+{protected_blocked - baseline_blocked}")
        st.metric("Complied", protected_complied, delta=f"{protected_complied - baseline_complied}")
        st.metric("Block Rate", f"{protected_blocked/total_p*100:.1f}%",
                  delta=f"+{(protected_blocked - baseline_blocked)/total_b*100:.1f}%")

    st.divider()

    # Per-category comparison
    st.subheader("Block Rate by OWASP Category")
    categories = sorted(set(i['category'] for i in baseline))
    before_rates = {}
    after_rates  = {}

    for cat in categories:
        b_items = [i for i in baseline  if i['category'] == cat]
        p_items = [i for i in protected if i['category'] == cat]
        before_rates[cat] = sum(1 for i in b_items if i['result'] == 'BLOCKED') / len(b_items) * 100 if b_items else 0
        after_rates[cat]  = sum(1 for i in p_items if i['result'] == 'BLOCKED') / len(p_items) * 100 if p_items else 0

    st.bar_chart({"Before": before_rates, "After": after_rates})

except Exception as e:
    st.error(f"Could not load benchmark data from GitHub: {str(e)}")

st.divider()

# ── Recent Logs ───────────────────────────────────────────────────────────────
st.subheader("Recent Logs")
sorted_items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
for item in sorted_items:
    st.write(f"`{item.get('timestamp', '')[:19]}` | **{item.get('attack_type')}** | Layer: `{item.get('layer_blocked')}` | Blocked: `{item.get('blocked')}`")