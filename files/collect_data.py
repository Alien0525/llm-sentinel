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
import datasets as _datasets
import os, re

ATTACK_TARGET  = 700   # ≥500 required
BENIGN_TARGET  = 700
OUTPUT_PATH    = "data/training_data.csv"
os.makedirs("data", exist_ok=True)

# Supplemental attack/jailbreak datasets to try (can be overridden with env var
# SUPPLEMENTAL_DATASETS as a comma-separated list). The script will attempt to
# load each dataset in order and keep samples until ATTACK_TARGET is reached.
DEFAULT_SUPPLEMENTAL = [
    "verazuo/jailbreak_prompts",
    "jasonml/jailbreak-prompts",    # common variant names (may not exist)
    "prompt-injection/jailbreaks",  # filler candidate — will be skipped if missing
]
sup_env = os.environ.get("SUPPLEMENTAL_DATASETS")
SUPPLEMENTAL_DATASETS = [s.strip() for s in sup_env.split(",") if s.strip()] if sup_env else DEFAULT_SUPPLEMENTAL

# ── 1. ATTACK samples ─────────────────────────────────────────────────────────
print("Downloading attack dataset (deepset/prompt-injections)...")
inj = load_dataset("deepset/prompt-injections", split="train")
inj_df = pd.DataFrame(inj).rename(columns={"text": "prompt", "label": "label"})
inj_df = inj_df[inj_df["label"] == 1][["prompt"]].copy()
inj_df["label"] = 1
inj_df["attack_type"] = "injection"
print(f"  Found {len(inj_df)} injection samples")

# ── 2. JAILBREAK samples (if available) ───────────────────────────────────────
# Try a list of supplemental jailbreak/attack datasets until we reach ATTACK_TARGET
def _extract_prompt_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with a single 'prompt' column extracted from common
    possible text fields.
    """
    candidates = ["prompt", "text", "instruction", "input", "content", "query"]
    for c in candidates:
        if c in df.columns:
            return df[[c]].rename(columns={c: "prompt"}).copy()
    # fallback: use the first string-like column
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            return df[[c]].rename(columns={c: "prompt"}).copy()
    # if nothing found, return empty DataFrame
    return pd.DataFrame(columns=["prompt"])


if len(inj_df) < ATTACK_TARGET and SUPPLEMENTAL_DATASETS:
    needed_total = ATTACK_TARGET - len(inj_df)
    print(f"Need {needed_total} more attack samples; trying supplemental datasets: {SUPPLEMENTAL_DATASETS}")
    for ds_id in SUPPLEMENTAL_DATASETS:
        if len(inj_df) >= ATTACK_TARGET:
            break
        print(f"  Trying supplemental dataset: {ds_id}")
        try:
            jb = load_dataset(ds_id, split="train")
            jb_df = pd.DataFrame(jb)
            jb_df = _extract_prompt_column(jb_df)
            if jb_df.empty:
                print(f"    Skipping {ds_id}: no prompt-like column found")
                continue
            jb_df["label"] = 1
            # label the attack type by dataset id (shortened)
            jb_df["attack_type"] = ds_id.split("/")[-1]
            needed = ATTACK_TARGET - len(inj_df)
            inj_df = pd.concat([inj_df, jb_df.head(needed)], ignore_index=True)
            print(f"    Added {len(jb_df.head(needed))} samples from {ds_id}; total attacks now {len(inj_df)}")
        except _datasets.exceptions.DatasetNotFoundError:
            print(f"    Dataset {ds_id} not found on the Hub — skipping")
        except Exception as e:
            print(f"    Warning: failed to load {ds_id}: {e} — skipping")

    if len(inj_df) < ATTACK_TARGET:
        print(f"  After supplemental attempts, attack samples: {len(inj_df)} (target {ATTACK_TARGET})")

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
