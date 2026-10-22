# PlacementPredict — ML System

A production-style **supervised classification** service that predicts whether a student will be
**placed** or not, built as a complete ML lifecycle: data → features → models → serving → report.

## Run it

```bash
pip install -r requirements.txt
python scripts/pipeline.py     # rebuild training data, models, metrics and every chart
python main.py                 # web app on http://127.0.0.1:5000
```

## What is inside

| Piece | Where |
|---|---|
| Raw dataset (50,000 students) | `data/placement_data.csv` |
| Pre-processing pipeline (impute → scale → encode) | `src/preprocessing.py` → `models/pipeline.pkl` |
| Training + metrics | `src/models.py` → `models/*.pkl`, `models/metrics.json` |
| Chart generation (M1–M4) | `src/charts.py` → `app/static/charts/` |
| Serving (form + JSON API) | `app/app.py`, `src/serving.py` |
| Standalone report | `src/report.py` → `reports/placement_report.html` |
| One-command retrain | `scripts/pipeline.py` |

## Endpoints

- `GET /` — dashboard (M1 lifecycle, KPIs, module map)
- `GET /lifecycle` — **M1** PlacementPredict as a system
- `GET /dataset`, `/eda`, `/preprocessing` — data pages with charts
- `GET /linear` — **M2** linear models (regularisation, logistic regression, encodings)
- `GET /trees` — **M3** tree models (splitting, pruning, bagging, boosting, importance)
- `GET /models`, `/evaluation`, `/comparison` — training run and metrics
- `GET /unsupervised` — **M4** clustering, PCA, t-SNE/UMAP, anomaly detection
- `GET /prediction` — interactive prediction form
- `POST /api/predict` — JSON prediction (single or batch)
- `GET /api/health` — service/artefact status
- `GET /report` — readable system report; **press Ctrl+S to save it**
- `GET /report/export` — download the same report as one self-contained HTML file

## Models trained

Logistic Regression (M2, deployed), Decision Tree, Random Forest, HistGradientBoosting (M3) —
all scored on the same stratified 80/20 hold-out set; the best-F1 model is the default artefact.
# PYTHON-PLACEMENT-PREDICTION-
