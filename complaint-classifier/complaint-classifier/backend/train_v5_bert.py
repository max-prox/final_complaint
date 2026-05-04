"""
train_v5_bert.py
----------------
The Ultimate Classifier Pipeline using Deep Learning NLP.
Replaces TF-IDF with HuggingFace Sentence Transformers (BERT).

What's new:
  1. Dense Semantic Embeddings (384-dimensional vector from all-MiniLM-L6-v2).
  2. Neural Network understands context, negations, and sentiment inherently.
  3. Fuses embeddings with our 20 hand-crafted features for domain-specific safety.
  4. Extreme regularization on classifiers to prevent overfitting in 404 dimensional space.

Run:
    python train_v5_bert.py

Outputs:
    models_v5/ensemble_v5.pkl
    models_v5/label_encoder_v5.pkl
    models_v5/feature_scaler_v5.pkl
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

from preprocessor_v2 import extract_features_batch

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

DATASET_PATH = "dataset_v5.csv"
MODELS_DIR   = "models_v5"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
LABEL_ORDER  = ["Critical", "High", "Medium", "Low", "No Issue"]

os.makedirs(MODELS_DIR, exist_ok=True)

def section(title):
    print(f"\n{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


# ─────────────────────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────────────────────

section("STEP 1 — Load Dataset")

df = pd.read_csv(DATASET_PATH)
print(f"  Loaded {len(df)} samples")

le = LabelEncoder()
le.fit(LABEL_ORDER)
df["label_enc"] = le.transform(df["label"])
raw_texts = df["text"].tolist()
y = df["label_enc"].values


# ─────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION: BERT + ENGINEERED
# ─────────────────────────────────────────────────────────────

section("STEP 2 — Feature Extraction (BERT + Engineered)")

print("  Loading all-MiniLM-L6-v2 Sentence Transformer...")
# A massive, high-accuracy BERT model (768 dimensions).
transformer = SentenceTransformer('all-mpnet-base-v2')

print("  Computing dense semantic embeddings (this may take a moment)...")
X_bert = transformer.encode(raw_texts, show_progress_bar=True)

print("  Extracting engineered features...")
X_eng = extract_features_batch(raw_texts)

# We fuse 384 dimensions of semantic meaning with 20 dimensions of strict rules
X_fused = np.hstack([X_bert, X_eng])

print(f"  Final feature shape: {X_fused.shape}")


# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

section("STEP 3 — Train / Test Split")

(X_train, X_test, y_train, y_test) = train_test_split(
    X_fused, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────────
# 4. BERT-OPTIMIZED ENSEMBLE
# ─────────────────────────────────────────────────────────────

section("STEP 4 — Deep Learning Optimized Ensemble")

# Model 1: Logistic Regression handles dense embeddings exceptionally well
lr = LogisticRegression(
    C=0.5, # Strong regularization
    max_iter=1000,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

# Model 2: Linear SVM (great for high-dim dense features)
svm = SVC(
    C=0.5,
    kernel='linear',
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

# Model 3: XGBoost to capture non-linear BERT + Engineered interactions
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    learning_rate=0.01, # very slow learning to prevent overfitting
    max_depth=4,        # shallow trees
    n_estimators=300,
    subsample=0.5,      # only use 50% data per tree
    colsample_bytree=0.5, # only use 50% features per tree
    random_state=RANDOM_STATE,
    eval_metric='mlogloss'
)

# Soft Voting Ensemble
ensemble_v5 = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('svm', svm),
        ('xgb', xgb)
    ],
    voting='soft'
)

print("  Training Ultimate Ensemble Model...")
ensemble_v5.fit(X_train_scaled, y_train)

y_pred = ensemble_v5.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n  Final Test Set Accuracy (BERT): {acc*100:.2f}%")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))


# ─────────────────────────────────────────────────────────────
# 5. STRESS TEST
# ─────────────────────────────────────────────────────────────

section("STEP 5 — Stress Test")

STRESS_TEST = [
    # Emotional language
    ("I am absolutely furious, this app ruined my entire day!",               "High"),
    ("This is the worst service I have ever experienced in my life.",          "High"),
    ("Nobody cares about customers anymore. Completely let down.",             "Medium"),
    ("Im so done with this platform, third time this week something broke",    "High"),
    # Billing / financial
    ("You charged me twice and nobody is picking up the phone",                "High"),
    ("My bank flagged a suspicious charge from your company",                  "High"),
    ("Still waiting for my refund from 3 weeks ago, very disappointed",        "Medium"),
    # Delivery
    ("My order never arrived and tracking shows delivered",                    "High"),
    # Healthcare
    ("Appointment was cancelled last minute with no notification",             "High"),
    # Casual
    ("its broken again lol nothing ever works here",                           "Medium"),
    ("cant do anything, page just spins forever",                              "High"),
    # Regulatory
    ("You shared my personal data without consent, GDPR violation",           "Critical"),
    # Positive framing
    ("Love the app but it keeps crashing on checkout which is really annoying","High"),
    # Ambiguous
    ("System said payment processed but my account shows no change",           "High"),
    # No Issue
    ("I absolutely love the new update, great job!",                           "No Issue"),
    ("How do I change my profile picture?",                                    "No Issue"),
]

def run_stress_test():
    print(f"\n  Model: V5 BERT Ensemble")
    print(f"  {'Complaint':<55} {'Expected':<10} {'Predicted':<10} {'OK?':<5} Conf")
    print("  " + "─" * 90)
    correct = 0
    for text, expected in STRESS_TEST:
        x_bert = transformer.encode([text])
        x_eng  = extract_features_batch([text])
        x_fuse = np.hstack([x_bert, x_eng])
        x_fuse_scaled = scaler.transform(x_fuse)

        proba = ensemble_v5.predict_proba(x_fuse_scaled)[0]
        pred  = le.classes_[np.argmax(proba)]
        conf  = round(proba.max() * 100)
        ok    = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"  {text[:54]:<55} {expected:<10} {pred:<10} {ok:<5} {conf}%")

    acc = correct / len(STRESS_TEST)
    print(f"\n  Stress-test accuracy: {correct}/{len(STRESS_TEST)} = {acc*100:.1f}%")

run_stress_test()


# ─────────────────────────────────────────────────────────────
# 6. SAVE MODELS
# ─────────────────────────────────────────────────────────────

section("STEP 6 — Save Models")

# Note: We do not save the transformer model itself as it's large and loaded dynamically
joblib.dump(ensemble_v5,  f"{MODELS_DIR}/ensemble_v5.pkl")
joblib.dump(le,     f"{MODELS_DIR}/label_encoder_v5.pkl")
joblib.dump(scaler, f"{MODELS_DIR}/feature_scaler_v5.pkl")

print(f"  ✅  {MODELS_DIR}/ensemble_v5.pkl")
print(f"  ✅  {MODELS_DIR}/label_encoder_v5.pkl")
print(f"  ✅  {MODELS_DIR}/feature_scaler_v5.pkl")
print(f"\n  Run: python predict_v5_bert.py \"your complaint here\"\n")
