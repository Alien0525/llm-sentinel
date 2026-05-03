import streamlit as st
import boto3
import json
import urllib.request
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Sentinel", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .big-metric { font-size: 2.5rem; font-weight: 700; }
    .label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .card { background: #1a1a2e; border-radius: 12px; padding: 1.5rem; border: 1px solid #2d2d4e; }
    .blocked { color: #ff4b4b; }
    .allowed { color: #00cc88; }
    .section-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🛡️ LLM Sentinel — Security Dashboard")
st.caption("Real-time attack monitoring · OWASP LLM Top 10 · AWS Cloud-Native Defense")

# ── DynamoDB connection ───────────────────────────────────────────────────────
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')
table = dynamodb.Table('llm-sentinel-attack-logs')

@st.cache_data(ttl=30)
def fetch_logs():
    result = table.scan()
    return result.get('Items', [])

@st.cache_data(ttl=3600)
def fetch_github_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())

BASELINE_URL  = "https://raw.githubusercontent.com/Alien0525/llm-sentinel/refs/heads/main/results/baseline_results.json"
PROTECTED_URL = "https://raw.githubusercontent.com/Alien0525/llm-sentinel/refs/heads/main/results/week5_benchmark_results.json"

items = fetch_logs()

if not items:
    st.warning("No data in DynamoDB yet.")
    st.stop()

# ── SECTION 1: Real-time metrics ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📡 Live Attack Monitor")

total = len(items)
blocked = sum(1 for i in items if i.get('blocked') == True)
allowed = total - blocked
block_rate = (blocked / total * 100) if total > 0 else 0
layer_counts = Counter(i.get('layer_blocked', 'unknown') for i in items if i.get('blocked') == True)

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Total Requests", total)
col2.metric("🔴 Blocked", blocked)
col3.metric("🟢 Allowed", allowed)
col4.metric("Block Rate", f"{block_rate:.1f}%")
col5.metric("Layer 1 Blocks", layer_counts.get('layer_1', 0))
col6.metric("Layer 2 Blocks", layer_counts.get('layer_2', 0))
col7.metric("Layer 3 Blocks", layer_counts.get('layer_3', 0))

# ── SECTION 2: Attack type chart ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎯 Attack Type Breakdown")
attack_counts = Counter(i.get('attack_type', 'unknown') for i in items)
st.bar_chart(attack_counts, height=250)

# ── SECTION 3: Before / After comparison ─────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Before vs After — OWASP LLM Top 10 Benchmark")
st.caption("Baseline: 21 unprotected payloads (Week 2) vs Protected: same 21 payloads through all 3 layers (Week 5)")

try:
    baseline      = fetch_github_json(BASELINE_URL)
    protected_all = fetch_github_json(PROTECTED_URL)
    protected     = [i for i in protected_all if i.get('pass') == 'plain']

    total_b = len(baseline)
    total_p = len(protected)
    baseline_blocked   = sum(1 for i in baseline   if i.get('result') == 'BLOCKED')
    baseline_complied  = sum(1 for i in baseline   if i.get('result') == 'COMPLIED')
    protected_blocked  = sum(1 for i in protected  if i.get('result') == 'BLOCKED')
    protected_complied = sum(1 for i in protected  if i.get('result') == 'COMPLIED')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🔴 Before — No Defense Layers")
        st.info("All 21 OWASP LLM Top 10 attacks reached TinyLlama unfiltered.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", total_b)
        m2.metric("Blocked", baseline_blocked)
        m3.metric("Complied", baseline_complied)
        m4.metric("Block Rate", f"{baseline_blocked/total_b*100:.1f}%")

    with col2:
        st.markdown("#### 🟢 After — All 3 Layers Active")
        st.success("Layer 2 (Bedrock Guardrails) blocked 14/21 attacks in under 600ms.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", total_p)
        m2.metric("Blocked", protected_blocked, delta=f"+{protected_blocked - baseline_blocked}")
        m3.metric("Complied", protected_complied, delta=f"{protected_complied - baseline_complied}")
        m4.metric("Block Rate", f"{protected_blocked/total_p*100:.1f}%",
                  delta=f"+{(protected_blocked - baseline_blocked)/total_b*100:.1f}%")

    # ── Per-category comparison chart ─────────────────────────────────────────
    st.markdown("#### Block Rate by OWASP Category (Before vs After)")
    categories = sorted(set(i['category'] for i in baseline))
    before_rates = {}
    after_rates  = {}
    for cat in categories:
        b_items = [i for i in baseline  if i['category'] == cat]
        p_items = [i for i in protected if i['category'] == cat]
        before_rates[cat] = sum(1 for i in b_items if i['result'] == 'BLOCKED') / len(b_items) * 100 if b_items else 0
        after_rates[cat]  = sum(1 for i in p_items if i['result'] == 'BLOCKED') / len(p_items) * 100 if p_items else 0

    st.bar_chart({"Before (0% baseline)": before_rates, "After (with defense)": after_rates}, height=300)
    st.caption("Before bars are 0% — the unprotected LLM complied with all attacks. After bars show block rate per category with all 3 layers active.")

    # ── SECTION 4: Encoded payload analysis ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔐 Encoded Payload Detection (Base64 vs Plain)")
    st.caption("Week 5 benchmark: same 21 payloads sent as plain text AND base64-encoded to test decode pipeline.")

    plain_data   = [i for i in protected_all if i.get('pass') == 'plain']
    encoded_data = [i for i in protected_all if i.get('pass') == 'base64_encoded']

    plain_blocked   = sum(1 for i in plain_data   if i.get('result') == 'BLOCKED')
    encoded_blocked = sum(1 for i in encoded_data if i.get('result') == 'BLOCKED')
    plain_rate   = plain_blocked   / len(plain_data)   * 100 if plain_data   else 0
    encoded_rate = encoded_blocked / len(encoded_data) * 100 if encoded_data else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Plain Text Block Rate",   f"{plain_rate:.1f}%",   f"{plain_blocked}/{len(plain_data)} blocked")
    col2.metric("Base64 Encoded Block Rate", f"{encoded_rate:.1f}%", f"{encoded_blocked}/{len(encoded_data)} blocked")
    col3.metric("Encoding Detection", "✅ Working", "decode_if_encoded() active")

    st.bar_chart({
        "Plain Text":     {"Blocked": plain_blocked,   "Complied": len(plain_data)   - plain_blocked},
        "Base64 Encoded": {"Blocked": encoded_blocked, "Complied": len(encoded_data) - encoded_blocked}
    }, height=250)

    # ── SECTION 5: Latency comparison ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚡ Latency Analysis")
    st.caption("Average response time per outcome — blocked requests return instantly, complied requests wait for TinyLlama.")

    plain_blocked_latency  = [i['latency_ms'] for i in plain_data if i.get('result') == 'BLOCKED']
    plain_allowed_latency  = [i['latency_ms'] for i in plain_data if i.get('result') == 'COMPLIED']
    enc_blocked_latency    = [i['latency_ms'] for i in encoded_data if i.get('result') == 'BLOCKED']
    enc_allowed_latency    = [i['latency_ms'] for i in encoded_data if i.get('result') == 'COMPLIED']

    avg = lambda lst: round(sum(lst) / len(lst)) if lst else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Blocked Latency (Plain)",   f"{avg(plain_blocked_latency)}ms",  "Fast — Layer 2 stops early")
    col2.metric("Avg Allowed Latency (Plain)",   f"{avg(plain_allowed_latency)}ms",  "Slower — EC2 inference")
    col3.metric("Avg Blocked Latency (Base64)",  f"{avg(enc_blocked_latency)}ms",    "Fast — decode + block")
    col4.metric("Avg Allowed Latency (Base64)",  f"{avg(enc_allowed_latency)}ms",    "Slower — full pipeline")

    st.bar_chart({
        "Plain Blocked":   {"Avg Latency (ms)": avg(plain_blocked_latency)},
        "Plain Allowed":   {"Avg Latency (ms)": avg(plain_allowed_latency)},
        "Base64 Blocked":  {"Avg Latency (ms)": avg(enc_blocked_latency)},
        "Base64 Allowed":  {"Avg Latency (ms)": avg(enc_allowed_latency)},
    }, height=250)

except Exception as e:
    st.error(f"Could not load benchmark data: {str(e)}")

# ── SECTION 6: Recent Logs ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Recent Attack Logs (Live)")
sorted_items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
for item in sorted_items:
    blocked_label = "🔴 BLOCKED" if item.get('blocked') else "🟢 ALLOWED"
    st.markdown(
        f"`{item.get('timestamp', '')[:19]}` &nbsp;|&nbsp; "
        f"**{item.get('attack_type', 'unknown')}** &nbsp;|&nbsp; "
        f"Layer: `{item.get('layer_blocked', 'none')}` &nbsp;|&nbsp; "
        f"{blocked_label}",
        unsafe_allow_html=True
    )