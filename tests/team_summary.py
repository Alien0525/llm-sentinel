"""
LLM Sentinel — Team Results Summary Generator
Varsha Roopesh | NYU Tandon | Spring 2026

Reads all result JSON files and prints a clean summary to share with the team.
Also generates a team_summary.md file you can paste into the group chat or GitHub.
"""

import json, glob, os, datetime

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "llm-sentinel")

files = sorted(
    glob.glob(os.path.join(HOME, "baseline_attack_log_*_corrected.json")) +
    glob.glob(os.path.join(REPO, "week5_benchmark_*.json")) +
    glob.glob(os.path.join(HOME, "week5_benchmark_*.json")) +
    glob.glob(os.path.join(REPO, "e2e_test_*.json"))
)

print("=" * 68)
print("LLM Sentinel — Team Results Summary")
print(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 68)

md_lines = [
    "# LLM Sentinel — Results Summary",
    f"_Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}_",
    "",
]

for filepath in files:
    name = os.path.basename(filepath)
    with open(filepath) as f:
        data = json.load(f)

    if not data:
        continue

    total = len(data)
    blocked  = sum(1 for r in data if r.get("result") == "BLOCKED"  or r.get("verdict") == "blocked")
    complied = sum(1 for r in data if r.get("result") == "COMPLIED" or (r.get("verdict") == "allowed" and not r.get("baseline")))
    errors   = sum(1 for r in data if r.get("result") == "ERROR"    or r.get("verdict") == "error")
    enc      = sum(1 for r in data if r.get("encoding_detected") == "base64")

    layer_counts = {}
    for r in data:
        l = r.get("blocked_by")
        if l:
            layer_counts[l] = layer_counts.get(l, 0) + 1

    lats = [r.get("latency_ms", 0) for r in data if (r.get("latency_ms") or 0) > 0]
    avg_lat = round(sum(lats)/len(lats)) if lats else 0
    block_rate = round(blocked/total*100) if total else 0

    print(f"\n  File    : {name}")
    print(f"  Total   : {total} requests")
    print(f"  BLOCKED : {blocked} ({block_rate}%)")
    print(f"  COMPLIED: {complied}")
    if enc:
        print(f"  Encoded : {enc} base64 payloads detected")
    if layer_counts:
        for layer, count in sorted(layer_counts.items()):
            print(f"    {layer}: {count} blocks")
    print(f"  Avg lat : {avg_lat}ms")

    md_lines += [
        f"## {name}",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total requests | {total} |",
        f"| BLOCKED | {blocked} ({block_rate}%) |",
        f"| COMPLIED | {complied} |",
    ]
    if enc:
        md_lines.append(f"| Base64 detected | {enc} |")
    for layer, count in sorted(layer_counts.items()):
        md_lines.append(f"| {layer} blocks | {count} |")
    md_lines += [f"| Avg latency | {avg_lat}ms |", ""]

# ── Key comparison table ──────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  BEFORE vs AFTER COMPARISON")
print("  " + "─" * 64)
print(f"  {'Metric':<35} {'Baseline':>12} {'Protected':>12}")
print("  " + "─" * 64)
rows = [
    ("Block rate (plain)",           "0%",    "86%"),
    ("Block rate (base64 encoded)",  "0%",    "95%"),
    ("Base64 encodings detected",    "N/A",   "21/21"),
    ("Layer 1 blocks",               "N/A",   "30"),
    ("Layer 2 blocks",               "N/A",   "0*"),
    ("Layer 3 blocks",               "N/A",   "8"),
    ("Avg blocked latency",          "N/A",   "~300ms"),
    ("Avg complied latency",         "5-10s", "~13s"),
    ("p50 latency (all requests)",   "N/A",   "297ms"),
    ("p95 latency (all requests)",   "N/A",   "17564ms"),
]
for metric, base, prot in rows:
    print(f"  {metric:<35} {base:>12} {prot:>12}")
print("  " + "─" * 64)
print("  * Layer 2 not triggered when Layer 1 catches first")

md_lines += [
    "## Before vs After Comparison",
    "| Metric | Baseline (unprotected) | Protected (all 3 layers) |",
    "|--------|------------------------|--------------------------|",
]
for metric, base, prot in rows:
    md_lines.append(f"| {metric} | {base} | {prot} |")
md_lines += ["", "_* Layer 2 not triggered when Layer 1 catches the request first_", ""]

# ── Save markdown ─────────────────────────────────────────────────────────────
md_path = os.path.join(REPO, "results", "team_summary.md")
os.makedirs(os.path.dirname(md_path), exist_ok=True)
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))

print(f"\n  Markdown saved: {md_path}")
print("  Share this file with the team or paste into your group chat.")
