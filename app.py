import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR   = os.path.join(BASE_DIR, "models_v5")

# Lazy load — model loads on first request, not at startup.
# This lets gunicorn bind the port instantly and avoids Render 503.
_predict_fn = None

def get_predict():
    global _predict_fn
    if _predict_fn is None:
        import numpy as np, joblib
        from sentence_transformers import SentenceTransformer
        from preprocessor_v2 import extract_features_batch

        transformer = SentenceTransformer("all-mpnet-base-v2")
        ensemble    = joblib.load(f"{MODELS_DIR}/ensemble_v5.pkl")
        le          = joblib.load(f"{MODELS_DIR}/label_encoder_v5.pkl")
        scaler      = joblib.load(f"{MODELS_DIR}/feature_scaler_v5.pkl")

        def predict(text: str) -> dict:
            if not text or not text.strip():
                return {"label": "Unknown", "confidence": 0.0, "all_scores": {}}
            x_bert  = transformer.encode([text])
            x_eng   = extract_features_batch([text])
            x_fused = np.hstack([x_bert, x_eng])
            x_scaled= scaler.transform(x_fused)
            proba   = ensemble.predict_proba(x_scaled)[0]
            idx     = int(proba.argmax())
            return {
                "label":      le.classes_[idx],
                "confidence": round(float(proba[idx]), 4),
                "all_scores": {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)},
            }

        _predict_fn = predict
    return _predict_fn


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    return jsonify(get_predict()(data["text"]))

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    full = os.path.join(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, path if os.path.exists(full) else "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
