# Complaint Classifier v5 — BERT Ensemble

Priority triage for support complaints using `all-mpnet-base-v2` Sentence Transformers + a Logistic Regression / SVM / XGBoost voting ensemble.

---

## Project Structure

```
complaint-classifier/
├── render.yaml                   ← Render deployment config
├── README.md
└── backend/
    ├── app.py                    ← Flask API + static file server
    ├── predict_v5_bert.py        ← Prediction logic
    ├── preprocessor_v2.py        ← Feature engineering
    ├── train_v5_bert.py          ← Training script (run locally)
    ├── dataset_v5.csv            ← Training data
    ├── requirements.txt
    ├── models_v5/
    │   ├── ensemble_v5.pkl
    │   ├── label_encoder_v5.pkl
    │   └── feature_scaler_v5.pkl
    └── frontend/
        └── index.html            ← Full SPA frontend
```

---

## Local Development

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

---

## Deploy to Render (GitHub)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/YOUR_USERNAME/complaint-classifier.git
git push -u origin main
```

> **Important:** The `models_v5/` folder contains `.pkl` files (~several MB).  
> Make sure they are **not** in `.gitignore` — they must be committed.

### 2. Create Render Web Service

1. Go to [render.com](https://render.com) → **New → Web Service**
2. Connect your GitHub repo
3. Render will auto-detect `render.yaml` and configure everything

**Manual settings if not using render.yaml:**

| Setting | Value |
|---|---|
| Root Directory | `backend` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1` |

### 3. Notes

- **First boot is slow (~60–90 seconds)** — Render downloads the `all-mpnet-base-v2` model (~420 MB) from HuggingFace. After that it's cached on the persistent disk.
- Use **at least 1 GB RAM** (Standard plan). The BERT model needs ~500 MB.
- Set a **persistent disk** at `/root/.cache` (5 GB) so the model isn't re-downloaded on every deploy — this is already in `render.yaml`.

---

## API

### `POST /api/classify`

```json
// Request
{ "text": "Server is completely down, urgent fix needed" }

// Response
{
  "label": "Critical",
  "confidence": 0.912,
  "all_scores": {
    "Critical": 0.912,
    "High": 0.063,
    "Medium": 0.018,
    "Low": 0.005,
    "No Issue": 0.002
  }
}
```

### `GET /api/health`

```json
{ "status": "ok" }
```

---

## Retrain

```bash
cd backend
python train_v5_bert.py
# Outputs: models_v5/ensemble_v5.pkl, label_encoder_v5.pkl, feature_scaler_v5.pkl
```
