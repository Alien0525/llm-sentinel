import streamlit as st
import boto3
import json
from collections import Counter
from datetime import datetime, timezone, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LLM Sentinel", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .big-metric { font-size: 2.5rem; font-weight: 700; }
    .label { font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("# 🛡️ LLM Sentinel — Security Dashboard")
st.caption("Real-time attack monitoring · OWASP LLM Top 10 · AWS Cloud-Native Defense")

# ── AWS clients ───────────────────────────────────────────────────────────────
dynamodb   = boto3.resource('dynamodb', region_name='us-east-2')
cloudwatch = boto3.client('cloudwatch', region_name='us-east-2')
table      = dynamodb.Table('llm-sentinel-attack-logs')

# ── Fetch all logs ────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_logs():
    result = table.scan()
    return result.get('Items', [])

# ── Fetch CloudWatch latency ──────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_latency(layer):
    response = cloudwatch.get_metric_statistics(
        Namespace='LLMSentinel',
        MetricName='LayerLatency',
        Dimensions=[{'Name': 'Layer', 'Value': layer}],
        StartTime=datetime.now(timezone.utc) - timedelta(hours=24),
        EndTime=datetime.now(timezone.utc),
        Period=86400,
        Statistics=['Average']
    )
    points = response.get('Datapoints', [])
    return round(points[0]['Average']) if points else 0

items = fetch_logs()

if not items:
    st.warning("No data in DynamoDB yet.")
    st.stop()

# ── Separate live vs benchmark ────────────────────────────────────────────────
benchmark_items = [i for i in items if i.get('owasp_id')]
live_items      = [i for i in items if not i.get('owasp_id')]

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📡 Live Requests", "📊 Benchmark Results", "⚡ Latency"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Requests
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📡 Live Attack Monitor")
    st.caption("All incoming requests logged in real-time via Lambda → DynamoDB. Refreshes every 30 seconds.")

    total      = len(live_items)
    blocked    = sum(1 for i in live_items if i.get('blocked') == True)
    allowed    = total - blocked
    block_rate = (blocked / total * 100) if total > 0 else 0
    cache_hits = sum(1 for i in live_items if i.get('cache_hit') == True)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Requests", total)
    col2.metric("🔴 Blocked",     blocked)
    col3.metric("🟢 Allowed",     allowed)
    col4.metric("Block Rate",     f"{block_rate:.1f}%")
    col5.metric("Cache Hits",     cache_hits)

    st.markdown("---")

    # Layer breakdown
    st.markdown("#### Layer Breakdown")
    layer_counts = Counter(i.get('layer_blocked', 'unknown') for i in live_items if i.get('blocked') == True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Layer 1 Blocks", layer_counts.get('Layer1', 0))
    col2.metric("Layer 2 Blocks", layer_counts.get('Layer2', 0))
    col3.metric("Layer 3 Blocks", layer_counts.get('Layer3', 0))
    col4.metric("Cache Blocks",   layer_counts.get('cache',  0))

    st.markdown("---")

    # Attack type chart
    st.markdown("#### Attack Types")
    attack_counts = Counter(i.get('attack_type', 'unknown') for i in live_items)
    st.bar_chart(attack_counts, height=250)

    st.markdown("---")

    # Encoding detection
    st.markdown("#### Encoding Detection")
    encoding_counts = Counter(i.get('encoding_detected', 'none') for i in live_items)
    col1, col2, col3 = st.columns(3)
    col1.metric("Base64 Detected", encoding_counts.get('base64', 0))
    col2.metric("Hex Detected",    encoding_counts.get('hex',    0))
    col3.metric("No Encoding",     encoding_counts.get('none',   0))

    st.markdown("---")

    # Recent logs
    st.markdown("#### Recent Logs")
    sorted_items = sorted(live_items, key=lambda x: x.get('timestamp', ''), reverse=True)[:20]
    for item in sorted_items:
        blocked_label = "🔴 BLOCKED" if item.get('blocked') else "🟢 ALLOWED"
        cache_label   = " 💾 cached" if item.get('cache_hit') else ""
        st.markdown(
            f"`{item.get('timestamp', '')[:19]}` &nbsp;|&nbsp; "
            f"**{item.get('attack_type', 'unknown')}** &nbsp;|&nbsp; "
            f"Layer: `{item.get('layer_blocked', 'none')}` &nbsp;|&nbsp; "
            f"Encoding: `{item.get('encoding_detected', 'none')}` &nbsp;|&nbsp; "
            f"{blocked_label}{cache_label}",
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Benchmark Results
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 OWASP LLM Top 10 Benchmark")
    st.caption("42 benchmark records from Week 5 — same payloads sent as plain text and base64 encoded.")

    if not benchmark_items:
        st.warning("No benchmark data found in DynamoDB. Make sure Varsha's 42 records have an `owasp_id` field.")
    else:
        # Plain vs encoded split
        plain_items   = [i for i in benchmark_items if i.get('encoding', 'none') == 'none']
        encoded_items = [i for i in benchmark_items if i.get('encoding', 'none') != 'none']

        # Summary
        total_b    = len(benchmark_items)
        b_blocked  = sum(1 for i in benchmark_items if i.get('result') == 'BLOCKED')
        b_complied = sum(1 for i in benchmark_items if i.get('result') == 'COMPLIED')
        b_rate     = (b_blocked / total_b * 100) if total_b > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Payloads", total_b)
        col2.metric("Blocked",        b_blocked)
        col3.metric("Complied",       b_complied)
        col4.metric("Block Rate",     f"{b_rate:.1f}%")

        st.markdown("---")

        # Plain vs Encoded comparison
        st.markdown("#### Plain Text vs Base64 Encoded")
        plain_blocked   = sum(1 for i in plain_items   if i.get('result') == 'BLOCKED')
        encoded_blocked = sum(1 for i in encoded_items if i.get('result') == 'BLOCKED')
        plain_rate   = (plain_blocked   / len(plain_items)   * 100) if plain_items   else 0
        encoded_rate = (encoded_blocked / len(encoded_items) * 100) if encoded_items else 0

        col1, col2 = st.columns(2)
        col1.metric("Plain Block Rate",   f"{plain_rate:.1f}%",   f"{plain_blocked}/{len(plain_items)} blocked")
        col2.metric("Base64 Block Rate",  f"{encoded_rate:.1f}%", f"{encoded_blocked}/{len(encoded_items)} blocked")

        st.bar_chart({
            "Plain Text":     {"Blocked": plain_blocked,   "Complied": len(plain_items)   - plain_blocked},
            "Base64 Encoded": {"Blocked": encoded_blocked, "Complied": len(encoded_items) - encoded_blocked}
        }, height=250)

        st.markdown("---")

        # Per OWASP category
        st.markdown("#### Block Rate by OWASP Category")
        categories = sorted(set(i.get('category', 'unknown') for i in plain_items))
        cat_blocked = {}
        cat_total   = {}
        for cat in categories:
            cat_items        = [i for i in plain_items if i.get('category') == cat]
            cat_total[cat]   = len(cat_items)
            cat_blocked[cat] = sum(1 for i in cat_items if i.get('result') == 'BLOCKED')

        cat_rates = {cat: round(cat_blocked[cat] / cat_total[cat] * 100) for cat in categories if cat_total[cat] > 0}
        st.bar_chart(cat_rates, height=300)

        st.markdown("---")

        # Avg latency per result
        st.markdown("#### Avg Latency by Outcome")
        blocked_latency = [int(i.get('latency_ms', 0)) for i in benchmark_items if i.get('result') == 'BLOCKED']
        allowed_latency = [int(i.get('latency_ms', 0)) for i in benchmark_items if i.get('result') == 'COMPLIED']
        avg = lambda lst: round(sum(lst) / len(lst)) if lst else 0

        col1, col2 = st.columns(2)
        col1.metric("Avg Blocked Latency", f"{avg(blocked_latency)}ms", "Fast — blocked early by Layer 2")
        col2.metric("Avg Allowed Latency", f"{avg(allowed_latency)}ms", "Slower — full EC2 inference")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Latency (CloudWatch)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### ⚡ Layer Latency (CloudWatch — Last 24 Hours)")
    st.caption("Average latency per layer from CloudWatch LLMSentinel namespace.")

    layers = ["Layer1_Classifier", "Layer2_Bedrock", "Layer3_Egress", "EC2_TinyLlama", "CacheLookup", "Decode"]

    latency_data = {}
    for layer in layers:
        latency_data[layer] = fetch_latency(layer)

    col1, col2, col3 = st.columns(3)
    col1.metric("Layer 1 (ML Classifier)", f"{latency_data['Layer1_Classifier']}ms")
    col2.metric("Layer 2 (Bedrock Input)", f"{latency_data['Layer2_Bedrock']}ms")
    col3.metric("Layer 3 (Egress)",        f"{latency_data['Layer3_Egress']}ms")

    col1, col2, col3 = st.columns(3)
    col1.metric("EC2 TinyLlama",  f"{latency_data['EC2_TinyLlama']}ms")
    col2.metric("Cache Lookup",   f"{latency_data['CacheLookup']}ms")
    col3.metric("Decode Step",    f"{latency_data['Decode']}ms")

    st.markdown("---")
    st.markdown("#### Latency Comparison")
    st.bar_chart(latency_data, height=300)
    st.caption("Lower is better. Cache hits return in <50ms. Blocked requests stop at Layer 2 (~400ms). Allowed requests wait for EC2 (~5000ms).")