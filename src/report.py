from base64 import b64encode
from datetime import date
from pathlib import Path

from src.config import CHARTS_DIR, PROJECT_ROOT

CSS = """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',Arial,sans-serif;background:#eef1f7;color:#1c2733;line-height:1.65}
.page{background:#fff;max-width:940px;margin:26px auto;padding:0 44px 60px;
      border:1px solid #e3e9f2;border-radius:18px;box-shadow:0 24px 60px -30px rgba(14,27,46,.45)}
header{background:linear-gradient(135deg,#0e1b2e,#17416b);color:#dbe7f8;padding:44px 44px 38px;
       margin:0 -44px 34px;border-radius:18px 18px 0 0}
header h1{font-size:29px;letter-spacing:-.02em;color:#fff}
header p{opacity:.85;margin-top:8px;font-size:14.5px}
h2{font-size:21px;margin:40px 0 8px;padding-bottom:9px;border-bottom:2px solid #e3e9f2;color:#0e1b2e}
h3{font-size:16px;margin:20px 0 6px;color:#14273f}
p,li{font-size:14.6px}
p{margin:9px 0}
ul,ol{margin:9px 0 9px 22px}
li{margin-bottom:5px}
.muted{color:#64748b;font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:16px 0}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:16px 0}
.card{background:#fff;border:1px solid #e3e9f2;border-radius:14px;padding:17px;
      box-shadow:0 10px 26px -18px rgba(14,27,46,.5)}
.card h4{color:#2563eb;font-size:12px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px}
img{max-width:100%;border:1px solid #e3e9f2;border-radius:9px;background:#f6f8fb}
figure{background:#f8fafd;border:1px solid #e3e9f2;border-radius:12px;padding:12px;margin:15px 0}
figcaption{color:#64748b;font-size:12.3px;text-align:center;margin-top:7px}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:13.4px}
th,td{padding:9px 12px;border:1px solid #e3e9f2;text-align:left}
th{background:#eef3fb;color:#14273f;font-weight:650}
tr:nth-child(even) td{background:#fafcff}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
.kpi{background:#fff;border:1px solid #e3e9f2;border-top:4px solid #2563eb;border-radius:12px;
     padding:15px;text-align:center}
.kpi:nth-child(2){border-top-color:#0d9488}.kpi:nth-child(3){border-top-color:#d97706}
.kpi:nth-child(4){border-top-color:#16a34a}
.kpi b{font-size:23px;color:#0e1b2e;display:block}
.kpi span{color:#64748b;font-size:12.3px}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:14px 0}
.step{background:#fff;border:1px solid #e3e9f2;border-radius:12px;padding:14px;font-size:13.6px}
.step b{color:#2563eb;display:block;margin-bottom:3px}
.step span{color:#64748b}
.callout{background:#eef4ff;border-left:4px solid #2563eb;border-radius:0 10px 10px 0;
         padding:13px 17px;margin:15px 0;font-size:14.2px}
.callout.amber{background:#fdf5e7;border-left-color:#d97706}
.callout.green{background:#ecf9f0;border-left-color:#16a34a}
.diagram{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:15px 0}
.node{background:#fff;border:1px solid #d3dcae;color:#14273f;border-radius:10px;padding:9px 13px;
      font-size:12.6px;font-weight:650;box-shadow:0 5px 0 -2px #dfe7f2}
.arrow{color:#94a3b8;font-size:17px;font-weight:700}
code{background:#eef3fb;border:1px solid #e3e9f2;padding:1px 6px;border-radius:6px;
     font-family:Consolas,monospace;font-size:12.6px;color:#1e3a8a}
footer{color:#64748b;font-size:12.2px;text-align:center;margin-top:44px;border-top:1px solid #e3e9f2;padding-top:16px}
@media print{body{background:#fff}.page{box-shadow:none;border:none;margin:0;border-radius:0}}
"""

KINDS = {
    "Logistic Regression": "Linear",
    "Decision Tree": "Tree",
    "Random Forest": "Bagging",
    "Gradient Boosting": "Boosting",
}


def img_tag(name, max_h=""):
    path = Path(CHARTS_DIR) / name
    if not path.exists():
        return ""
    data = b64encode(path.read_bytes()).decode()
    style = f"style='max-height:{max_h}px;object-fit:contain'" if max_h else ""
    return f"<img {style} src='data:image/png;base64,{data}' alt='{name}'>"


def metric_rows(metrics, best):
    rows = ["<tr><th>Model</th><th>Type</th><th>Accuracy</th><th>Precision</th>"
            "<th>Recall</th><th>F1</th><th>ROC AUC</th></tr>"]
    for name, m in metrics.items():
        star = " &#9679; deployed" if name == best else ""
        rows.append(
            f"<tr><td><b>{name}</b>{star}</td><td>{KINDS.get(name, '—')}</td>"
            f"<td>{m['accuracy']:.2f}</td><td>{m['precision']:.2f}</td>"
            f"<td>{m['recall']:.2f}</td><td><b>{m['f1']:.3f}</b></td>"
            f"<td>{m['roc_auc']:.3f}</td></tr>")
    return "\n".join(rows)


def build_report(metrics, summary, best_name):
    rows = summary.get("rows", 0)
    cols = summary.get("columns", 0)
    nulls = summary.get("nulls", 0)
    rate = summary.get("placed_rate", 0.65)
    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>PlacementPredict — ML System Report</title><style>{CSS}</style></head><body><div class='page'>

<header>
<h1>PlacementPredict — Machine Learning System Report</h1>
<p>Predicting student placement outcomes &middot; supervised classification &middot;
data &rarr; model &rarr; serve &rarr; monitor<br>
Generated {date.today().strftime('%d %B %Y')} &middot; M1 system &middot; M2 linear &middot; M3 trees &middot; M4 unsupervised</p>
</header>

<h2>1. Executive summary</h2>
<p>PlacementPredict answers one question: <b>will this student be placed?</b> A profile of ~30 inputs
passes through a frozen pre-processing pipeline, is scored by a trained classifier, and returns an
outcome plus a probability in milliseconds. This report walks the complete machine-learning
lifecycle in plain language — suitable for anyone to read, no ML background required.</p>
<div class='kpis'>
<div class='kpi'><b>{rows:,}</b><span>Student records</span></div>
<div class='kpi'><b>{cols}</b><span>Raw features</span></div>
<div class='kpi'><b>{rate:.0%}</b><span>Placement rate</span></div>
<div class='kpi'><b>{best_name}</b><span>Deployed model (best F1)</span></div>
</div>

<h2>2. PlacementPredict as a system <span class='muted'>(M1)</span></h2>
<p>Every ML product follows the same lifecycle; each stage has a concrete home in this codebase:</p>
<div class='diagram'>
<span class='node'>Raw data</span><span class='arrow'>&rarr;</span>
<span class='node'>Clean</span><span class='arrow'>&rarr;</span>
<span class='node'>Features</span><span class='arrow'>&rarr;</span>
<span class='node'>Train set</span><span class='arrow'>&rarr;</span>
<span class='node'>Model</span><span class='arrow'>&rarr;</span>
<span class='node'>Artefact</span><span class='arrow'>&rarr;</span>
<span class='node'>Serve</span><span class='arrow'>&rarr;</span>
<span class='node'>Monitor</span><span class='arrow'>&rarr;</span>
<span class='node'>Retrain</span>
</div>
<table>
<tr><th>Lifecycle stage</th><th>Where it lives in code</th></tr>
<tr><td>Raw data</td><td><code>data/placement_data.csv</code> — {rows:,} student profiles</td></tr>
<tr><td>Cleaning &amp; feature engineering</td><td><code>src/preprocessing.py</code> — impute, standardise, encode</td></tr>
<tr><td>Training set</td><td><code>data/preprocessed_train.csv</code> — stratified 80/20 split</td></tr>
<tr><td>Training script</td><td><code>src/models.py</code> — four classifiers + metrics</td></tr>
<tr><td>Model registry</td><td><code>models/*.pkl</code>, <code>models/default_model.pkl</code></td></tr>
<tr><td>Serving framework</td><td><code>app/app.py</code>, <code>src/serving.py</code> — form + JSON API</td></tr>
<tr><td>Monitoring</td><td><code>/api/health</code> — artefact state, model, metrics</td></tr>
<tr><td>Retraining loop</td><td><code>scripts/pipeline.py</code> — one command rebuilds everything</td></tr>
</table>
<div class='callout'><b>What kind of learning?</b> <b>Supervised classification</b> — labels exist
(Placed / Not Placed) and the model learns profile &rarr; outcome. It is not unsupervised (no label
discovery) and not reinforcement learning (no agent, no reward). Clustering in M4 only <i>explores</i>
the same data.</div>
<div class='callout'><b>One prediction, end to end:</b> <code>HTTP POST</code> &rarr; feature lookup
&rarr; <code>pipeline.transform()</code> &rarr; <code>predict_proba()</code> &rarr; JSON.
Batch prediction runs the identical path over many rows — one code path, so single and batch
scoring can never disagree.</div>
<div class='callout amber'><b>The training–serving boundary.</b> The bug that breaks the most
production systems is <b>training-serving skew</b> — features computed differently at serve time than
at train time. Feature stores keep one canonical definition; here the fitted
<code>pipeline.pkl</code> is exactly that: fit once on training data, reused verbatim everywhere.</div>
<div class='callout green'><b>Course scope:</b> classical ML and ML engineering live here (M1–M4).
Deep learning moves to DLAC, generative models to GALE, LLMs to NLPL.</div>

<h2>3. Data &amp; exploratory analysis <span class='muted'>(M1)</span></h2>
<p>The dataset holds <b>{rows:,} students</b> and {cols} attributes. <b>{nulls:,} cells</b> are
missing (five skill columns) and are repaired in preprocessing. The target is imbalanced about 2:1
toward placed students — which is why F1 and AUC, not accuracy alone, decide the winner.</p>
<div class='grid'>
<div class='card'><h4>Outcome distribution</h4>{img_tag('target_dist.png')}</div>
<div class='card'><h4>CGPA distribution</h4>{img_tag('cgpa_hist.png')}</div>
<div class='card'><h4>CGPA vs aptitude</h4>{img_tag('score_scatter.png')}</div>
<div class='card'><h4>CGPA by outcome</h4>{img_tag('cgpa_box.png')}</div>
<div class='card'><h4>Outcome by stream</h4>{img_tag('stream_target.png')}</div>
<div class='card'><h4>Correlation matrix</h4>{img_tag('corr_matrix.png')}</div>
</div>
<div class='callout'><b>Insight:</b> CGPA, aptitude and coding scores cleanly separate placed from
not-placed students; semester grades correlate strongly with CGPA, so the model relies on a compact
set of academic signals.</div>

<h2>4. Data preparation pipeline <span class='muted'>(M1)</span></h2>
<div class='steps'>
<div class='step'><b>1 · Clean</b><span>Drop ID, IsAnomaly and Salary Package (label leakage).</span></div>
<div class='step'><b>2 · Impute</b><span>Median for numerics, most-frequent for categories.</span></div>
<div class='step'><b>3 · Scale</b><span>StandardScaler so gradient descent converges.</span></div>
<div class='step'><b>4 · Encode</b><span>One-hot for categories, ordinal for the two tiers.</span></div>
<div class='step'><b>5 · Split</b><span>Stratified 80/20, random_state = 42.</span></div>
<div class='step'><b>6 · Freeze</b><span>Fitted transformer pickled as pipeline.pkl.</span></div>
</div>
<div class='grid'>
<div class='card'><h4>Missing cells by column</h4>{img_tag('missing.png')}</div>
<div class='card'><h4>Before / after standardisation</h4>{img_tag('scaling.png')}</div>
</div>

<h2>5. Linear models <span class='muted'>(M2)</span></h2>
<p><b>Regularisation:</b> ridge (L2) shrinks weights smoothly, lasso (L1) drives weak weights to
exactly zero, elastic net combines both. The geometry explains the sparsity — the loss contour
meets the L1 diamond at a corner, and a corner means a zero weight.</p>
<div class='grid'>
<div class='card'><h4>Why L1 produces sparsity</h4>{img_tag('regularization.png')}</div>
<div class='card'><h4>Sigmoid &amp; cross-entropy loss</h4>{img_tag('sigmoid.png')}</div>
</div>
<p><b>Logistic regression</b> computes P(placed) = &sigma;(w&middot;x + b): a hyperplane boundary plus
a sigmoid, trained by gradient descent on cross-entropy with ridge regularisation. Coefficients are
log-odds, so every score can be explained to a stakeholder. Feature scaling matters for gradient
descent — not for the trees in M3. Multinomial (softmax) regression generalises the same idea to
k classes; this task is binary.</p>
<figure>{img_tag('lr_coefs.png', 400)}<figcaption>Top coefficients — the linear model's reading of the data</figcaption></figure>
<div class='callout'><b>Categorical features &amp; missing values:</b> one-hot for unordered
categories, ordinal for ordered ones (target and embedding encodings are advanced alternatives);
deletion, mean/median, model-based imputation and missingness flags were compared — median /
most-frequent is what production uses here.</div>

<h2>6. Tree-based models <span class='muted'>(M3)</span></h2>
<p><b>Decision trees</b> split greedily on Gini impurity and stop at depth/leaf limits;
cost-complexity pruning trims branches whose removal barely changes the error. A single deep tree
overfits — the classic bias&ndash;variance curve below.</p>
<figure>{img_tag('bias_variance.png', 360)}<figcaption>Bias&ndash;variance trade-off as tree depth grows</figcaption></figure>
<p><b>Random forests</b> bag bootstrap samples with feature subsampling to cut variance and give an
out-of-bag error for free. <b>Boosting</b> fits each new tree to the residual gradient of the loss;
histogram-based learners with leaf-wise growth (XGBoost / LightGBM — and sklearn's
HistGradientBoosting used here) are the modern industrial workhorses on tabular data. Feature
importance comes in three flavours: Gini importance, permutation importance, and SHAP values
(preview level).</p>
<figure>{img_tag('feat_importance.png', 380)}<figcaption>Random-forest feature importance</figcaption></figure>
<div class='callout amber'><b>Why trees win on tabular data:</b> no scaling, native handling of
missing values and mixed types, cheap tuning, non-linear interactions for free. Neural networks win
on images, text and audio — DLAC / NLPL territory.</div>

<h2>7. Evaluation &amp; comparison <span class='muted'>(M1–M3)</span></h2>
<table>{metric_rows(metrics, best_name)}</table>
<div class='grid'>
<div class='card'><h4>ROC curves</h4>{img_tag('roc_curves.png')}</div>
<div class='card'><h4>Confusion matrix — {best_name}</h4>{img_tag('confusion.png')}</div>
</div>
<figure>{img_tag('model_compare.png', 340)}<figcaption>Accuracy &middot; F1 &middot; AUC by model</figcaption></figure>
<div class='callout'><b>Reading the table:</b> accuracy is the share of correct calls; precision is
how reliable a &ldquo;placed&rdquo; call is; recall is how many truly placed students we catch; F1
balances both; AUC measures ranking quality. All four models exceed 90% accuracy —
<b>{best_name}</b> is deployed for the best F1 combined with interpretability.</div>

<h2>8. Serving the prediction <span class='muted'>(M1)</span></h2>
<div class='diagram'>
<span class='node'>POST /api/predict</span><span class='arrow'>&rarr;</span>
<span class='node'>feature lookup</span><span class='arrow'>&rarr;</span>
<span class='node'>pipeline.transform()</span><span class='arrow'>&rarr;</span>
<span class='node'>predict_proba()</span><span class='arrow'>&rarr;</span>
<span class='node'>JSON</span>
</div>
<ul>
<li><code>/prediction</code> — interactive form with a live result card;</li>
<li><code>/api/predict</code> — JSON in / JSON out for other systems;</li>
<li><code>/api/health</code> — artefact state and metrics for monitoring;</li>
<li><code>/report/export</code> — this report as a self-contained HTML file.</li>
</ul>
<p>The frozen pipeline means the transform at serve time is byte-for-byte the transform used in
training — the practical defence against training&ndash;serving skew.</p>

<h2>9. Unsupervised learning <span class='muted'>(M4)</span></h2>
<p>No labels here — structure, segments and anomalies. K-means with the elbow method and silhouette
score finds three coherent student segments; hierarchical clustering reads merge structure from a
dendrogram; DBSCAN grows clusters by density and marks noise points as outliers; PCA compresses
features (always standardise first); t-SNE and UMAP visualise high dimensions — but distances in
their spaces are not meaningful. Isolation forest and one-class SVM detect anomalies without
labels, with autoencoders covered in DLAC.</p>
<div class='grid'>
<div class='card'><h4>K-means elbow</h4>{img_tag('elbow.png')}</div>
<div class='card'><h4>Silhouette by k</h4>{img_tag('silhouette.png')}</div>
<div class='card'><h4>Student segments (3-D)</h4>{img_tag('kmeans_3d.png')}</div>
<div class='card'><h4>DBSCAN clusters &amp; outliers</h4>{img_tag('dbscan.png')}</div>
<div class='card'><h4>PCA variance &amp; dendrogram</h4>{img_tag('dim_reduction.png')}</div>
<div class='card'><h4>Isolation forest anomalies</h4>{img_tag('anomaly.png')}</div>
</div>
<div class='callout green'><b>The production trick:</b> fit k-means, append each student's
segment id as one more feature, and let the supervised classifier use group membership as extra
signal — unsupervised structure feeding a supervised model.</div>

<h2>10. Conclusions, limitations &amp; reproduction</h2>
<ul>
<li>Academic performance dominates placement outcomes; skills (aptitude, coding, interviews) add signal.</li>
<li>Predictions are a <b>decision aid</b> — they cannot capture company demand, interview luck or market swings.</li>
<li>The ~2:1 imbalance means accuracy alone would mislead; F1 and AUC are tracked instead.</li>
<li>Retraining on fresh cohorts each semester keeps the model honest as profiles drift.</li>
</ul>
<div class='callout green'><b>Reproduce everything:</b><br>
<code>pip install -r requirements.txt</code><br>
<code>python scripts/pipeline.py</code> — rebuilds training data, models, metrics and every chart.<br>
<code>python app/app.py</code> — serves the web app at http://localhost:5000.<br>
In the web app press <b>Ctrl+S</b> to save this report, or use the export button.</div>

<footer>PlacementPredict &middot; supervised binary classification &middot;
{date.today().strftime('%B %Y')}</footer>
</div></body></html>"""
    return html


def export_report(metrics, summary, best_name):
    out = PROJECT_ROOT / "reports" / "placement_report.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_report(metrics, summary, best_name), encoding="utf-8")
    return out
