import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = "data/full_dataset.csv"
MODEL_DIR = "model"

BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pkl")
TRAINING_REPORT_PATH = "data/training_report.txt"

RANDOM_STATE = 42


def load_existing_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"{DATASET_PATH} not found. Run train_hf.py first."
        )

    df = pd.read_csv(DATASET_PATH)

    df = df.dropna(subset=["text", "label", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 10]
    df = df.drop_duplicates(subset=["text"])

    print("\nLoaded dataset:")
    print(df["label"].value_counts())

    print("\nIntent distribution:")
    print(df["intent"].value_counts())

    return df


def train_binary_classifier(df):
    print("\n==============================")
    print("Training binary attack classifier")
    print("==============================")

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
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
                solver="liblinear"
            )
        )
    ])

    param_grid = {
        "classifier__C": [0.1, 1.0, 5.0]
    }

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        scoring="f1_macro"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nBest parameters:")
    print(grid.best_params_)

    print("\nBinary accuracy:")
    print(accuracy)

    print("\nBinary classification report:")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    joblib.dump(best_model, BINARY_MODEL_PATH)

    with open(TRAINING_REPORT_PATH, "w") as f:
        f.write("Binary Classifier Training Report\n")
        f.write("================================\n\n")
        f.write(f"Best Parameters: {grid.best_params_}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    print(f"\nSaved binary model to: {BINARY_MODEL_PATH}")


def train_intent_classifier(df):
    print("\n==============================")
    print("Training intent classifier")
    print("==============================")

    X = df["text"]
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
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
            OneVsRestClassifier(
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear"
                )
            )
        )
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("\nIntent accuracy:")
    print(accuracy)

    print("\nIntent classification report:")
    print(report)

    joblib.dump(pipeline, INTENT_MODEL_PATH)

    with open(TRAINING_REPORT_PATH, "a") as f:
        f.write("\n\nIntent Classifier Training Report\n")
        f.write("=================================\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nSaved intent model to: {INTENT_MODEL_PATH}")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_existing_dataset()

    train_binary_classifier(df)

    train_intent_classifier(df)

    print("\n==============================")
    print("Compatible retraining complete")
    print("==============================")
    print(f"Binary model: {BINARY_MODEL_PATH}")
    print(f"Intent model: {INTENT_MODEL_PATH}")
    print(f"Training report: {TRAINING_REPORT_PATH}")