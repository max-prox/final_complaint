"""
predict_v5_bert.py
------------------
CLI for the ultimate V5 complaint classifier using Deep Learning NLP.

Usage:
    python predict_v5_bert.py "Server is completely down, urgent fix needed"
    python predict_v5_bert.py --demo
"""

import sys
import argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer
from preprocessor_v2 import extract_features_batch

# ─────────────────────────────────────────────────────────────
# LOAD MODELS (lazy, cached)
# ─────────────────────────────────────────────────────────────

import os
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_v5")
_cache = {}

def _load():
    if _cache:
        return _cache
    try:
        # Load transformer (this takes a moment if not cached)
        _cache["transformer"] = SentenceTransformer('all-mpnet-base-v2')
        
        _cache["ensemble"] = joblib.load(f"{MODELS_DIR}/ensemble_v5.pkl")
        _cache["le"]     = joblib.load(f"{MODELS_DIR}/label_encoder_v5.pkl")
        _cache["scaler"] = joblib.load(f"{MODELS_DIR}/feature_scaler_v5.pkl")
    except FileNotFoundError as e:
        print(f"\n  [!]  Model not found: {e}")
        print("     Run:  python train_v5_bert.py  first\n")
        sys.exit(1)
    return _cache


# ─────────────────────────────────────────────────────────────
# CORE PREDICT
# ─────────────────────────────────────────────────────────────

def predict_complaint_v5(text: str) -> dict:
    if not text or not text.strip():
        return {"label": "Unknown", "confidence": 0.0, "all_scores": {}}

    m = _load()
    
    # 1. BERT Dense Embedding
    x_bert = m["transformer"].encode([text])
    
    # 2. Engineered Features
    x_eng = extract_features_batch([text])
    
    # 3. Fuse and Scale
    x_fuse = np.hstack([x_bert, x_eng])
    x_fuse_scaled = m["scaler"].transform(x_fuse)

    # Predict
    proba      = m["ensemble"].predict_proba(x_fuse_scaled)[0]
    class_idx  = int(np.argmax(proba))
    label      = m["le"].classes_[class_idx]
    confidence = float(round(proba[class_idx], 4))
    all_scores = {m["le"].classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "label"            : label,
        "confidence"       : confidence,
        "all_scores"       : all_scores,
    }


# ─────────────────────────────────────────────────────────────
# TERMINAL DISPLAY
# ─────────────────────────────────────────────────────────────

COLORS = {
    "Critical": "\033[91m", "High": "\033[93m",
    "Medium":   "\033[94m", "Low":  "\033[92m",
    "RESET": "\033[0m",     "DIM":  "\033[2m",  "BOLD": "\033[1m",
}
ICONS = {"Critical": "🔴", "High": "🟠", "Medium": "🔵", "Low": "🟢"}


def _bar(val, width=20):
    filled = int(val * width)
    return "█" * filled + "░" * (width - filled)


def print_result(text: str, result: dict):
    label = result["label"]
    conf  = result["confidence"]
    icon  = ICONS.get(label, "⚪")
    col   = COLORS.get(label, "")
    rst   = COLORS["RESET"]
    dim   = COLORS["DIM"]

    print()
    print(f"{dim}  ┌─ Complaint Classifier v5 (BERT) ─────────────────────┐{rst}")
    print(f"  │ Input       : {text[:52]!r}")
    print(f"  │ Priority    : {icon} {col}{label}{rst}  {dim}({int(conf*100)}% confidence){rst}")
    print(f"  │ Confidence  : {_bar(conf)} {conf:.1%}")
    print(f"  │ All scores  :")
    for cls, score in sorted(result["all_scores"].items(), key=lambda x: -x[1]):
        c = COLORS.get(cls, ""); i = ICONS.get(cls, "")
        print(f"  │   {i} {cls:<10} {score:.3f}  {_bar(score, 10)}")

    print(f"{dim}  └────────────────────────────────────────────────────────┘{rst}")
    print()


# ─────────────────────────────────────────────────────────────
# DEMO COMPLAINTS
# ─────────────────────────────────────────────────────────────

DEMO = [
    "Server is completely down, urgent fix needed",
    "App is slow sometimes",
    "I am absolutely furious, this app ruined my entire day",
    "You shared my personal data without consent, GDPR violation",
    "Love the app but it keeps crashing on checkout",
    "This app is so easy to use, thank you!",
]


# ─────────────────────────────────────────────────────────────
# MAIN CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎯 Complaint Classifier v5 — Deep Learning AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("complaint", nargs="?", help="Complaint text to classify")
    parser.add_argument("--demo", action="store_true", help="Run on built-in examples")
    args = parser.parse_args()

    if args.demo:
        print(f"\n{COLORS['BOLD']}  🎯  Complaint Classifier v5 — Demo{COLORS['RESET']}\n")
        # Pre-load to avoid repeating "Loading..." prints
        _load()
        for c in DEMO:
            r = predict_complaint_v5(c)
            print_result(c, r)
        return

    if not args.complaint:
        parser.print_help()
        sys.exit(1)

    text = args.complaint.strip()
    r    = predict_complaint_v5(text)
    print_result(text, r)


if __name__ == "__main__":
    main()
