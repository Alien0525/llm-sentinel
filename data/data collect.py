"""
Task: Training Data Collection
LLM Sentinel — Week 2 | Owner: Krisha

Downloads HuggingFace datasets and builds a balanced labeled CSV
with ≥500 attack samples and ≥500 benign samples.

Datasets used:
  - Attack:  deepset/prompt-injections  (label=1 rows)
  - Benign:  allenai/prosocial-dialog   (clean conversational prompts)

Run:
    pip install datasets scikit-learn pandas
    python collect_data.py
"""

import pandas as pd
from datasets import load_dataset
import os, re

ATTACK_TARGET  = 700   # ≥500 required
BENIGN_TARGET  = 700
OUTPUT_PATH    = "data/training_data.csv"
os.makedirs("data", exist_ok=True)

# ── 1. ATTACK samples ─────────────────────────────────────────────────────────
print("Downloading attack dataset (deepset/prompt-injections)...")
inj = load_dataset("deepset/prompt-injections", split="train")
inj_df = pd.DataFrame(inj).rename(columns={"text": "prompt", "label": "label"})
inj_df = inj_df[inj_df["label"] == 1][["prompt"]].copy()
inj_df["label"] = 1
inj_df["attack_type"] = "injection"
print(f"  Found {len(inj_df)} injection samples")

# ── 2. JAILBREAK samples (supplemental, best-effort) ─────────────────────────
# Try known public jailbreak datasets in priority order to top up attack samples.
# Each is wrapped individually — if a dataset is renamed, gated, or removed from
# the Hub the script skips it and moves on rather than crashing.
SUPPLEMENTAL_DATASETS = [
    # (repo_id, prompt_column)
    ("jackhhao/jailbreak-classification", "prompt"),
    ("rubend18/ChatGPT-Jailbreak-Prompts", "Prompt"),
    ("TrustAIRLab/in-the-wild-jailbreak-prompts", "prompt"),
]

if len(inj_df) < ATTACK_TARGET:
    for repo_id, col in SUPPLEMENTAL_DATASETS:
        if len(inj_df) >= ATTACK_TARGET:
            break
        needed = ATTACK_TARGET - len(inj_df)
        print(f"Downloading supplemental dataset ({repo_id})...")
        try:
            jb = load_dataset(repo_id, split="train")
            jb_df = pd.DataFrame(jb)
            if col not in jb_df.columns:
                print(f"  ⚠️  Column '{col}' not found in {repo_id} — skipping")
                continue
            jb_df = jb_df[[col]].rename(columns={col: "prompt"}).copy()
            jb_df["label"] = 1
            jb_df["attack_type"] = "jailbreak"
            inj_df = pd.concat([inj_df, jb_df.head(needed)], ignore_index=True)
            print(f"  ✅ Supplemented to {len(inj_df)} attack samples")
        except Exception as e:
            print(f"  ⚠️  Could not load {repo_id}: {e} — skipping")

    if len(inj_df) < ATTACK_TARGET:
        print(f"  ⚠️  Only {len(inj_df)} attack samples collected (target: {ATTACK_TARGET}).")
        print(f"      Proceeding — add more datasets to SUPPLEMENTAL_DATASETS to reach target.")

# ── 3. BENIGN samples ─────────────────────────────────────────────────────────
print("Downloading benign dataset (allenai/prosocial-dialog)...")
pro = load_dataset("allenai/prosocial-dialog", split=f"train[:{BENIGN_TARGET * 3}]")
pro_df = pd.DataFrame(pro)[["context"]].rename(columns={"context": "prompt"}).copy()
pro_df["label"] = 0
pro_df["attack_type"] = "benign"
# Filter out very short or malformed entries
pro_df = pro_df[pro_df["prompt"].str.strip().str.len() > 20].copy()
print(f"  Found {len(pro_df)} benign samples after filtering")

# ── 4. Balance and save ───────────────────────────────────────────────────────
attack_final = inj_df.sample(n=min(ATTACK_TARGET, len(inj_df)), random_state=42)
benign_final = pro_df.sample(n=min(BENIGN_TARGET, len(pro_df)), random_state=42)

df = pd.concat([attack_final, benign_final], ignore_index=True)
df["prompt"] = df["prompt"].astype(str).str.strip()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

df.to_csv(OUTPUT_PATH, index=False)

vc = df["label"].value_counts()
print(f"\n✅ Saved to {OUTPUT_PATH}")
print(f"   Total: {len(df)} | Attack: {vc[1]} | Benign: {vc[0]}")
print(f"   Attack types: {dict(df[df.label==1]['attack_type'].value_counts())}")
print("\nSample rows:")
print(df.sample(5)[["prompt", "label", "attack_type"]].to_string(index=False))