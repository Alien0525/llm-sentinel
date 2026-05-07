import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATA_CANDIDATES = [
    "data/full_dataset.csv",
    "data/training_data.csv",
    "data/dataset.csv"
]

MODEL_PATH = "model/model.pkl"
RANDOM_STATE = 42


def find_dataset():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No dataset found. Expected one of: "
        + ", ".join(DATA_CANDIDATES)
    )


def normalize_columns(df):
    print("Original columns:", df.columns.tolist())

    if "text" not in df.columns:
        possible_text_cols = ["prompt", "Prompt", "instruction", "query", "content"]
        for col in possible_text_cols:
            if col in df.columns:
                df = df.rename(columns={col: "text"})
                break

    if "label" not in df.columns:
        possible_label_cols = ["class", "target", "is_attack", "category"]
        for col in possible_label_cols:
            if col in df.columns:
                df = df.rename(columns={col: "label"})
                break

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain text and label columns.")

    df = df[["text", "label"]].copy()

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    label_map = {
        "1": "attack",
        "0": "benign",
        "malicious": "attack",
        "harmful": "attack",
        "jailbreak": "attack",
        "prompt_injection": "attack",
        "safe": "benign",
        "normal": "benign"
    }

    df["label"] = df["label"].replace(label_map)

    df = df[df["label"].isin(["attack", "benign"])]
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset=["text"])

    return df


def train():
    dataset_path = find_dataset()
    print(f"Using dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df = normalize_columns(df)

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=30000,
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                C=5.0
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nAccuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved compatible model to: {MODEL_PATH}")


if __name__ == "__main__":
    train()