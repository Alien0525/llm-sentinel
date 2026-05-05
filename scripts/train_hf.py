import os
import pandas as pd
import joblib

from datasets import load_dataset

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data"
MODEL_DIR = "model"

FULL_DATASET_PATH = os.path.join(DATA_DIR, "full_dataset.csv")
BINARY_MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pkl")
TRAINING_REPORT_PATH = os.path.join(DATA_DIR, "training_report.txt")

# Strong but manageable training size for laptop
MAX_PER_CLASS = 100000

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def safe_load_dataset(dataset_name, config_name=None, split="train"):
    """
    Loads a HuggingFace dataset.

    Since you already logged in using huggingface-cli login,
    we do not manually pass HF_TOKEN here.
    """

    try:
        print(f"\nLoading dataset: {dataset_name}")

        if config_name:
            ds = load_dataset(dataset_name, config_name, split=split)
        else:
            ds = load_dataset(dataset_name, split=split)

        print(f"Loaded: {dataset_name}")
        return ds

    except Exception as e:
        print(f"\nCould not load dataset: {dataset_name}")
        print(f"Reason: {e}")
        print("Skipping this dataset and continuing...\n")
        return None


def clean_text_dataframe(df):
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()

    # Remove empty or very short prompts
    df = df[df["text"].str.len() > 10]

    # Remove duplicate prompts
    df = df.drop_duplicates(subset=["text"])

    return df


# ============================================================
# DATASET 1: NECENT
# ============================================================

def load_necent():
    """
    Dataset:
    Necent/llm-jailbreak-prompt-injection-dataset

    This dataset contains metadata columns that help us label prompts
    as attack or benign.
    """

    ds = safe_load_dataset(
        "Necent/llm-jailbreak-prompt-injection-dataset",
        split="train"
    )

    if ds is None:
        return pd.DataFrame(columns=["text", "label", "intent"])

    df = pd.DataFrame(ds)

    print("Necent columns:", df.columns.tolist())

    if "prompt" not in df.columns:
        raise ValueError("Necent dataset does not contain a prompt column.")

    df = df.dropna(subset=["prompt"]).copy()
    df["text"] = df["prompt"].astype(str)

    def make_binary_label(row):
        is_dangerous = str(row.get("is_dangerous", "")).lower()
        prompt_harmful = str(row.get("prompt_harmful", "")).lower()
        prompt_adversarial = str(row.get("prompt_adversarial", "")).lower()

        if (
            is_dangerous in ["true", "1", "yes"]
            or prompt_harmful in ["true", "1", "yes"]
            or prompt_adversarial in ["true", "1", "yes"]
        ):
            return "attack"

        return "benign"

    def make_intent(row):
        attack_technique = str(row.get("attack_technique", "")).lower()
        category = str(row.get("category", "")).lower()
        prompt_type = str(row.get("prompt_type", "")).lower()

        combined = f"{attack_technique} {category} {prompt_type}"

        if "injection" in combined:
            return "prompt_injection"

        if "jailbreak" in combined:
            return "jailbreak"

        if "pii" in combined or "privacy" in combined or "data" in combined:
            return "pii_extraction"

        if "harmful" in combined or "dangerous" in combined:
            return "harmful_request"

        # Fallback
        if row["label"] == "attack":
            return "jailbreak"

        return "benign"

    df["label"] = df.apply(make_binary_label, axis=1)
    df["intent"] = df.apply(make_intent, axis=1)

    df = df[["text", "label", "intent"]].copy()
    df = clean_text_dataframe(df)

    print("Necent label distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# DATASET 2: JAILBREAKBENCH
# ============================================================

def load_jbb():
    """
    Dataset:
    JailbreakBench/JBB-Behaviors

    Important:
    This dataset does not have a train split.
    It has two splits:
    - harmful
    - benign

    We load both and label them correctly.
    """

    harmful_ds = safe_load_dataset(
        "JailbreakBench/JBB-Behaviors",
        config_name="behaviors",
        split="harmful"
    )

    benign_ds = safe_load_dataset(
        "JailbreakBench/JBB-Behaviors",
        config_name="behaviors",
        split="benign"
    )

    frames = []

    if harmful_ds is not None:
        harmful_df = pd.DataFrame(harmful_ds)
        print("JBB harmful columns:", harmful_df.columns.tolist())

        text_col = None
        for col in ["Behavior", "behavior", "Goal", "goal", "prompt", "text", "instruction"]:
            if col in harmful_df.columns:
                text_col = col
                break

        if text_col is None:
            text_col = harmful_df.columns[0]

        harmful_df = harmful_df[[text_col]].copy()
        harmful_df.columns = ["text"]
        harmful_df["label"] = "attack"
        harmful_df["intent"] = "jailbreak"
        harmful_df = clean_text_dataframe(harmful_df)

        frames.append(harmful_df)

    if benign_ds is not None:
        benign_df = pd.DataFrame(benign_ds)
        print("JBB benign columns:", benign_df.columns.tolist())

        text_col = None
        for col in ["Behavior", "behavior", "Goal", "goal", "prompt", "text", "instruction"]:
            if col in benign_df.columns:
                text_col = col
                break

        if text_col is None:
            text_col = benign_df.columns[0]

        benign_df = benign_df[[text_col]].copy()
        benign_df.columns = ["text"]
        benign_df["label"] = "benign"
        benign_df["intent"] = "benign"
        benign_df = clean_text_dataframe(benign_df)

        frames.append(benign_df)

    if not frames:
        return pd.DataFrame(columns=["text", "label", "intent"])

    df = pd.concat(frames, ignore_index=True)

    print("JBB label distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# DATASET 3: CENTREPOURLASECURITEIA
# ============================================================

def load_centre():
    """
    Dataset:
    centrepourlasecuriteia/jailbreak-dataset

    This dataset may be gated.
    If access is not available, the script skips it safely.
    """

    ds = safe_load_dataset(
        "centrepourlasecuriteia/jailbreak-dataset",
        split="train"
    )

    if ds is None:
        return pd.DataFrame(columns=["text", "label", "intent"])

    df = pd.DataFrame(ds)

    print("Centre dataset columns:", df.columns.tolist())

    possible_text_cols = [
        "prompt",
        "text",
        "jailbreak",
        "question",
        "instruction"
    ]

    text_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        text_col = df.columns[0]

    df = df[[text_col]].copy()
    df.columns = ["text"]

    df["label"] = "attack"
    df["intent"] = "jailbreak"

    df = clean_text_dataframe(df)

    print("Centre label distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# DATASET 4: WILDJAILBREAK BENIGN DATA
# ============================================================

def load_benign():
    """
    Dataset:
    allenai/wildjailbreak

    This dataset may be gated.
    If access is not available, the script skips it safely.

    Necent and JBB already provide benign data,
    so this is useful but not mandatory.
    """

    ds = safe_load_dataset(
        "allenai/wildjailbreak",
        split="train"
    )

    if ds is None:
        return pd.DataFrame(columns=["text", "label", "intent"])

    df = pd.DataFrame(ds)

    print("WildJailbreak columns:", df.columns.tolist())

    possible_text_cols = [
        "prompt",
        "text",
        "adversarial",
        "vanilla",
        "instruction"
    ]

    text_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        text_col = df.columns[0]

    label_candidates = [
        "label",
        "data_type",
        "category",
        "type"
    ]

    label_col = None

    for col in label_candidates:
        if col in df.columns:
            label_col = col
            break

    if label_col:
        print(f"Using benign filter column: {label_col}")
        print(df[label_col].value_counts().head(20))

        benign_keywords = [
            "benign",
            "safe",
            "vanilla",
            "normal"
        ]

        df = df[
            df[label_col]
            .astype(str)
            .str.lower()
            .apply(lambda x: any(word in x for word in benign_keywords))
        ]

    df = df[[text_col]].copy()
    df.columns = ["text"]

    df["label"] = "benign"
    df["intent"] = "benign"

    df = clean_text_dataframe(df)

    print("Benign label distribution:")
    print(df["label"].value_counts())

    return df


# ============================================================
# BUILD FINAL DATASET
# ============================================================

def build_dataset():
    print("\n==============================")
    print("Loading HuggingFace datasets...")
    print("==============================")

    df_necent = load_necent()
    df_jbb = load_jbb()
    df_centre = load_centre()
    df_benign = load_benign()

    df = pd.concat(
        [df_necent, df_jbb, df_centre, df_benign],
        ignore_index=True
    )

    df = clean_text_dataframe(df)

    print("\n==============================")
    print("Before balancing")
    print("==============================")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise ValueError(
            "The dataset only has one class. "
            "You need both attack and benign samples."
        )

    balanced_parts = []

    for label in df["label"].unique():
        part = df[df["label"] == label]

        if len(part) > MAX_PER_CLASS:
            part = part.sample(MAX_PER_CLASS, random_state=RANDOM_STATE)

        balanced_parts.append(part)

    df = pd.concat(balanced_parts, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print("\n==============================")
    print("After balancing")
    print("==============================")
    print(df["label"].value_counts())

    print("\n==============================")
    print("Intent distribution")
    print("==============================")
    print(df["intent"].value_counts())

    df.to_csv(FULL_DATASET_PATH, index=False)

    print(f"\nSaved full dataset to: {FULL_DATASET_PATH}")

    return df


# ============================================================
# TRAIN BINARY CLASSIFIER
# ============================================================

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

    print("\nBinary classifier accuracy:")
    print(accuracy)

    print("\nClassification report:")
    print(report)

    print("\nConfusion matrix:")
    print(matrix)

    joblib.dump(best_model, BINARY_MODEL_PATH)

    print(f"\nSaved binary classifier to: {BINARY_MODEL_PATH}")

    with open(TRAINING_REPORT_PATH, "w") as f:
        f.write("Binary Classifier Training Report\n")
        f.write("================================\n\n")
        f.write(f"Best Parameters: {grid.best_params_}\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(matrix))

    print(f"Saved training report to: {TRAINING_REPORT_PATH}")

    return best_model


# ============================================================
# TRAIN INTENT CLASSIFIER
# ============================================================

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

    print("\nIntent classifier accuracy:")
    print(accuracy)

    print("\nIntent classification report:")
    print(report)

    joblib.dump(pipeline, INTENT_MODEL_PATH)

    print(f"\nSaved intent classifier to: {INTENT_MODEL_PATH}")

    with open(TRAINING_REPORT_PATH, "a") as f:
        f.write("\n\nIntent Classifier Training Report\n")
        f.write("=================================\n\n")
        f.write(f"Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    return pipeline


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    ensure_dirs()

    df = build_dataset()

    train_binary_classifier(df)

    train_intent_classifier(df)

    print("\n==============================")
    print("Training complete")
    print("==============================")
    print(f"Dataset saved at: {FULL_DATASET_PATH}")
    print(f"Binary model saved at: {BINARY_MODEL_PATH}")
    print(f"Intent model saved at: {INTENT_MODEL_PATH}")
    print(f"Training report saved at: {TRAINING_REPORT_PATH}")