import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from predict_v5_bert import predict_complaint_v5

app = Flask(__name__)
CORS(app)

# ─── FRONTEND DIR ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# ─── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data["text"]
    result = predict_complaint_v5(text)
    return jsonify(result)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# ─── SERVE FRONTEND ────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    full = os.path.join(FRONTEND_DIR, path)
    if os.path.exists(full):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

# ─── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🔄 Loading BERT model on startup...")
    predict_complaint_v5("warmup")
    print("✅ Model ready. Starting server...")
    app.run(host="0.0.0.0", port=port, debug=False)
