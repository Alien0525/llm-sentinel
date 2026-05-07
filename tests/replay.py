"""
LLM Sentinel — Benchmark Replay Script
Varsha Roopesh | NYU Tandon | Spring 2026

Replays saved benchmark JSON logs through the live endpoint so results
appear on Aman's dashboard. Open the dashboard before running this.

Usage:
    python3 llm_sentinel_replay.py

Replays:
  - baseline_attack_log (unprotected, baseline=True)
  - week5_benchmark (protected, all layers active)
"""

import requests
import json
import time
import glob
import os

ENDPOINT = "https://rzraagkgb9.execute-api.us-east-2.amazonaws.com/dev/prompt"
TIMEOUT  = 45
DELAY    = 1.2   # seconds between requests — keeps dashboard readable live

# ── Find result files ─────────────────────────────────────────────────────────
HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "llm-sentinel")

def find_file(pattern_list):
    for pattern in pattern_list:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]  # most recent
    return None

baseline_file = find_file([
    os.path.join(HOME, "baseline_attack_log_*_corrected.json"),
    os.path.join(HOME, "baseline_attack_log_*.json"),
])

week5_file = find_file([
    os.path.join(REPO, "week5_benchmark_*.json"),
    os.path.join(HOME, "week5_benchmark_*.json"),
])

print("=" * 68)
print("LLM Sentinel — Dashboard Replay")
print(f"Endpoint : {ENDPOINT}")
print(f"Baseline : {baseline_file}")
print(f"Week5    : {week5_file}")
print("=" * 68)
print()
print("  Open https://llm-sentinel-frontend.s3.us-east-2.amazonaws.com/index.html")
print("  then press ENTER to start replay...")
input()


def replay(label, filepath, baseline_mode):
    print(f"\n{'─'*68}")
    print(f"Replaying: {label}")
    print(f"File     : {filepath}")
    print(f"Mode     : {'UNPROTECTED (baseline)' if baseline_mode else 'PROTECTED (all layers)'}")
    print(f"{'─'*68}")

    with open(filepath) as f:
        records = json.load(f)

    # Deduplicate — only send unique payloads
    seen = set()
    unique = []
    for r in records:
        payload = r.get("payload_preview") or r.get("payload") or ""
        if payload and payload not in seen:
            seen.add(payload)
            unique.append(r)

    print(f"Sending {len(unique)} unique payloads (skipping duplicates)...")
    print()

    blocked = allowed = errors = 0

    for i, r in enumerate(unique):
        payload = r.get("payload_preview") or r.get("payload") or ""
        if not payload:
            continue

        # Truncated previews end with "..." — use as-is, still valid attack signal
        try:
            resp = requests.post(
                ENDPOINT,
                json={"prompt": payload, "baseline": baseline_mode},
                timeout=TIMEOUT
            )
            body = resp.json()
            verdict  = body.get("verdict", "error")
            layer    = body.get("blocked_by") or "none"
            enc      = body.get("encoding_detected") or ""
            category = r.get("category", r.get("owasp_id", ""))[:28]

            if verdict == "blocked":
                blocked += 1
                tag = f"BLOCKED  [{layer}]"
            elif verdict in ("allowed", "baseline"):
                allowed += 1
                tag = "ALLOWED"
            else:
                errors += 1
                tag = f"ERROR ({verdict})"

            enc_tag = f" enc={enc}" if enc else ""
            print(f"  [{i+1:2d}/{len(unique)}] {tag:18s}{enc_tag:12s} {category}")

        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{len(unique)}] ERROR    {str(e)[:50]}")

        time.sleep(DELAY)

    total = blocked + allowed + errors
    rate  = round(blocked / total * 100) if total else 0
    print(f"\n  Done. Blocked={blocked} Allowed={allowed} Errors={errors} Block rate={rate}%")


# ── Run replays ───────────────────────────────────────────────────────────────

if baseline_file:
    replay(
        "Baseline (unprotected TinyLlama — no defenses)",
        baseline_file,
        baseline_mode=True
    )
    print("\n  Baseline replay done. Check dashboard — then press ENTER for Week 5 protected run...")
    input()
else:
    print("Baseline file not found — skipping.")

if week5_file:
    replay(
        "Week 5 Benchmark (all 3 layers active)",
        week5_file,
        baseline_mode=False
    )
else:
    print("Week 5 file not found — skipping.")

print("\n" + "=" * 68)
print("Replay complete. Dashboard should show both runs.")
print("Take a screenshot of the dashboard now for your report.")
print("=" * 68)
