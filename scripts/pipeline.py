import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.config import CHARTS_DIR, METRICS, PROJECT_ROOT, RAW_DATA
from src.charts import (
    anomaly_chart, bias_variance_chart, cluster_charts, compare_chart,
    confusion_chart, dim_reduction_chart, eda_charts, missing_chart,
    model_charts, regularization_chart, scaling_chart, sigmoid_chart,
)
from src.models import best_model_name, run_training
from src.preprocessing import run_placement_pipeline

SUMMARY = PROJECT_ROOT / "models" / "summary.json"


def save_summary():
    df = pd.read_csv(RAW_DATA)
    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "nulls": int(df.isnull().sum().sum()),
        "target_counts": df["PlacementStatus"].value_counts().to_dict(),
        "placed_rate": float(round(df["PlacementStatus"].mean(), 4)),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print("== 1/7 preprocessing ==")
    _, Xtr, Xte, ytr, yte = run_placement_pipeline()

    print("== 2/7 training ==")
    metrics, models, best_model = run_training()

    print("== 3/7 EDA charts ==")
    eda_charts()

    print("== 4/7 model charts ==")
    model_charts(models, Xte, yte)
    compare_chart(metrics)
    confusion_chart(metrics, best_model_name(metrics))

    print("== 5/7 unsupervised charts ==")
    cluster_charts()

    print("== 6/7 concept charts (M2 / M3 / M4) ==")
    sigmoid_chart()
    regularization_chart()
    scaling_chart()
    missing_chart()
    bias_variance_chart()
    dim_reduction_chart()
    anomaly_chart()

    print("== 7/7 summary ==")
    summary = save_summary()
    print("done:", METRICS, "|", summary)
    for path in sorted(CHARTS_DIR.glob("*.png")):
        print("  chart:", path.name, path.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()