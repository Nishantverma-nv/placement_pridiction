import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request, send_file

from src.config import (
    CATEGORICALS, METRICS, NUMERICS, PROJECT_ROOT, SUMMARY,
)
from src.data.load_data import get_summary, load_data
from src.models import best_model_name
from src.report import build_report
from src.serving import init_artifacts, predict, ready, state

app = Flask(__name__)

# ---------------------------------------------------------------- helpers

def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _raw():
    if not hasattr(app, "_raw_df"):
        app._raw_df = load_data()
    return app._raw_df


def context(**extra):
    summary = _read_json(SUMMARY, {})
    counts = summary.get("target_counts", {})
    summary["target_counts"] = {int(k): int(v) for k, v in counts.items()}
    ctx = {
        "active": None,
        "summary": summary,
        "metrics": _read_json(METRICS, {}),
    }
    ctx["best"] = best_model_name(ctx["metrics"]) if ctx["metrics"] else "—"
    ctx.update(extra)
    return ctx


# form specification for the prediction page
NUMERIC_FIELDS = [
    ("CGPA", "CGPA (0–10)", 0, 10, 0.01, 7.5),
    ("AttendancePercent", "Attendance %", 0, 100, 0.1, 80),
    ("AptitudeTestScore", "Aptitude test score", 0, 100, 0.1, 70),
    ("CodingTestScore", "Coding test score", 0, 100, 0.1, 62),
    ("MockInterviewScore", "Mock interview score", 0, 100, 0.1, 55),
    ("SoftSkillsRating", "Soft-skills rating (1–5)", 1, 5, 0.1, 3.6),
    ("Internships", "Internships", 0, 10, 1, 1),
    ("Projects", "Projects", 0, 20, 1, 3),
    ("Certifications", "Certifications", 0, 20, 1, 2),
    ("Workshops", "Workshops", 0, 20, 1, 1),
    ("Publications", "Publications", 0, 10, 1, 0),
    ("ExtraCurricular", "Extra-curricular (0–10)", 0, 10, 1, 2),
]
SEM_FIELDS = [(f"SGPA_Sem{i}", f"Semester {i} SGPA", 0, 10, 0.01, None) for i in range(1, 9)]


def _options(col):
    return sorted(_raw()[col].dropna().astype(str).unique().tolist())


def select_fields():
    return [(c, c, _options(c)) for c in CATEGORICALS]


# ---------------------------------------------------------------- pages

@app.route("/")
@app.route("/home")
def home():
    from pathlib import Path
    charts = list((Path(app.static_folder) / "charts").glob("*.png"))
    return render_template("home.html", **context(chart_count=len(charts)))


@app.route("/lifecycle")
def lifecycle():
    return render_template("lifecycle.html", **context(active="lifecycle"))


@app.route("/dataset")
def dataset():
    df = _raw()
    summary = get_summary(df)
    rows = df.head(8).where(df.head(8).notna(), None).to_dict(orient="records")
    return render_template(
        "dataset.html",
        **context(active="dataset", summary=summary,
                  columns=summary["dtypes"], nulls=summary["missing"], rows=rows),
    )


@app.route("/eda")
def eda():
    return render_template("eda.html", **context(active="eda"))


@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html", **context(
        active="preprocessing",
        n_num=len(NUMERICS), n_cat=len(CATEGORICALS),
    ))


@app.route("/linear")
def linear():
    return render_template("linear.html", **context(active="linear"))


@app.route("/trees")
def trees():
    return render_template("trees.html", **context(active="trees"))


@app.route("/models")
def models_page():
    return render_template("models.html", **context(active="models"))


@app.route("/evaluation")
def evaluation():
    metrics = context()["metrics"]
    best = context()["best"]
    return render_template("evaluation.html", **context(
        active="evaluation", metrics=metrics, best=best,
        best_metrics=metrics.get(best, {}),
    ))


@app.route("/comparison")
def comparison():
    return render_template("comparison.html", **context(active="comparison"))


@app.route("/unsupervised")
def unsupervised():
    return render_template("unsupervised.html", **context(active="unsupervised"))


@app.route("/prediction")
def prediction():
    return render_template(
        "prediction.html",
        **context(active="prediction",
                  numeric_fields=NUMERIC_FIELDS,
                  sem_fields=SEM_FIELDS,
                  select_fields=select_fields()),
    )


@app.route("/report")
def report():
    return render_template("report.html", **context(active="report"))


@app.route("/report/export")
def report_export():
    ctx = context()
    html = build_report(ctx["metrics"], ctx["summary"], ctx["best"])
    path = PROJECT_ROOT / "reports" / "placement_report.html"
    path.parent.mkdir(exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return send_file(path, as_attachment=True,
                     download_name="PlacementPredict_Report.html")


# ---------------------------------------------------------------- API

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    if not data:
        return jsonify({"error": "empty payload"}), 400
    try:
        return jsonify(predict(data))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def api_health():
    ok = ready() or init_artifacts()
    bucket = state()
    metrics = _read_json(METRICS, {})
    best = best_model_name(metrics) if metrics else None
    return jsonify({
        "status": "ok" if ok else "degraded",
        "model_loaded": ok,
        "default_model": bucket.get("model").__class__.__name__ if bucket.get("model") else None,
        "best_f1_model": best,
        "best_f1": metrics.get(best, {}).get("f1") if best else None,
        "records": (bucket.get("summary") or {}).get("rows"),
        "endpoints": ["/api/predict", "/api/health", "/report/export"],
    })


init_artifacts()


if __name__ == "__main__":
    app.run(debug=True)
