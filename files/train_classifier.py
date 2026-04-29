"""
Task: TF-IDF Classifier Training
LLM Sentinel — Week 2 | Owner: Krisha

Trains a scikit-learn TF-IDF + Logistic Regression pipeline,
evaluates accuracy, and saves model.pkl for containerization.

Run:
    python train_classifier.py
    # Produces: model/model.pkl  model/metrics.json
"""

import pandas as pd, pickle, json, os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay,
)
import numpy as np

DATA_PATH   = "data/training_data.csv"
MODEL_PATH  = "model/model.pkl"
METRICS_PATH = "model/metrics.json"
os.makedirs("model", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples | attack: {(df.label==1).sum()} | benign: {(df.label==0).sum()}")

X = df["prompt"].tolist()
y = df["label"].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build pipeline ────────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),      # unigrams, bigrams, trigrams
        max_features=10_000,
        sublinear_tf=True,       # log-scale TF
        min_df=1,
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
    )),
    ("clf", LogisticRegression(
        C=5.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )),
])

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining...")
pipeline.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

report_dict = classification_report(
    y_test, y_pred, target_names=["benign", "attack"], output_dict=True
)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")

metrics = {
    "accuracy":         float(report_dict["accuracy"]),
    "attack_precision": float(report_dict["attack"]["precision"]),
    "attack_recall":    float(report_dict["attack"]["recall"]),
    "attack_f1":        float(report_dict["attack"]["f1-score"]),
    "benign_precision": float(report_dict["benign"]["precision"]),
    "benign_recall":    float(report_dict["benign"]["recall"]),
    "benign_f1":        float(report_dict["benign"]["f1-score"]),
    "roc_auc":          float(roc_auc_score(y_test, y_prob)),
    "cv_f1_mean":       float(cv_scores.mean()),
    "cv_f1_std":        float(cv_scores.std()),
    "train_size":       len(X_train),
    "test_size":        len(X_test),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
}

# ── Print report ──────────────────────────────────────────────────────────────
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["benign", "attack"]))
print(f"ROC-AUC:        {metrics['roc_auc']:.4f}")
print(f"CV F1 (5-fold): {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
print(f"  [TN={metrics['confusion_matrix'][0][0]}  FP={metrics['confusion_matrix'][0][1]}]")
print(f"  [FN={metrics['confusion_matrix'][1][0]}  TP={metrics['confusion_matrix'][1][1]}]")

# ── Show top features ─────────────────────────────────────────────────────────
feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
coefs = pipeline.named_steps["clf"].coef_[0]
top_attack = [feature_names[i] for i in coefs.argsort()[-15:][::-1]]
top_benign = [feature_names[i] for i in coefs.argsort()[:15]]
print(f"\nTop attack indicators:  {top_attack}")
print(f"Top benign indicators:  {top_benign}")

# ── Save model and metrics ────────────────────────────────────────────────────
with open(MODEL_PATH, "wb") as f:
    pickle.dump(pipeline, f)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ model.pkl  → {MODEL_PATH}")
print(f"✅ metrics    → {METRICS_PATH}")
