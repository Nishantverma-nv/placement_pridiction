import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import CHARTS_DIR, PALETTE

BLUE, TEAL, AMBER, GREEN, VIOLET, SLATE = PALETTE

plt.rcParams.update({
    "figure.facecolor": "#f6f8fb",
    "axes.facecolor": "#ffffff",
    "savefig.facecolor": "#f6f8fb",
    "axes.edgecolor": "#e2e8f0",
    "axes.grid": True,
    "grid.color": "#eef1f5",
    "grid.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": "#1e293b",
    "font.family": "DejaVu Sans",
})
CMAP = LinearSegmentedColormap.from_list("prof", ["#ffffff", TEAL, BLUE])


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def eda_charts():
    df = pd.read_csv("data/placement_data.csv")
    d = df.copy()
    d["PlacementStatus"] = d["PlacementStatus"].map({1: "Placed", 0: "Not Placed"})

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    counts = d["PlacementStatus"].value_counts()
    ax.bar(counts.index, counts.values, color=[GREEN, SLATE], width=0.55)
    ax.bar_label(ax.containers[0], fmt="%d", padding=3)
    ax.set_title("Placement Outcome Distribution")
    ax.set_ylabel("Students")
    _save(fig, "target_dist.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(df["CGPA"], bins=28, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(df["CGPA"].median(), color=AMBER, ls="--", lw=1.6, label=f"median {df['CGPA'].median():.2f}")
    ax.legend(frameon=False)
    ax.set_title("CGPA Distribution of Students")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Students")
    _save(fig, "cgpa_hist.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for label, color in [("Placed", GREEN), ("Not Placed", SLATE)]:
        sub = d[d["PlacementStatus"] == label]
        ax.scatter(sub["CGPA"], sub["AptitudeTestScore"], s=12, c=color, alpha=0.5, label=label)
    ax.set_title("CGPA vs Aptitude Test Score")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Aptitude Test Score")
    ax.legend(frameon=False, title="Outcome")
    _save(fig, "score_scatter.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    d.boxplot(column="CGPA", by="PlacementStatus", ax=ax, showmeans=True,
              color=dict(boxes=BLUE, whiskers=BLUE, medians=AMBER, caps=BLUE),
              meanprops=dict(marker="D", markerfacecolor=GREEN, markeredgecolor="white"))
    ax.set_title("CGPA by Placement Outcome")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("CGPA")
    fig.suptitle("")
    _save(fig, "cgpa_box.png")

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ct = pd.crosstab(d["Stream"], d["PlacementStatus"]).reindex(["CS", "IT", "ECE", "EE", "Mechanical", "Civil"])
    ct.plot(kind="bar", ax=ax, color=[SLATE, GREEN], width=0.72)
    ax.set_title("Placement Outcome by Stream")
    ax.set_xlabel("Stream")
    ax.set_ylabel("Students")
    ax.legend(title="Outcome", frameon=False)
    for c in ax.containers:
        ax.bar_label(c, fontsize=8)
    _save(fig, "stream_target.png")

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    g = df["Gender"].value_counts()
    ax.pie(g.values, labels=g.index, colors=[TEAL, SLATE], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor="white"), autopct="%1.0f%%")
    ax.set_title("Gender Distribution")
    _save(fig, "gender_pie.png")

    num = df.select_dtypes("number").drop(columns=["StudentID"]).corr()
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    im = ax.imshow(num, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num.columns)), num.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(num.columns)), num.columns, fontsize=8)
    ax.grid(False)
    plt.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    ax.set_title("Correlation Matrix — Numerical Features")
    _save(fig, "corr_matrix.png")


def model_charts(models, Xte, yte):
    Xte = Xte.drop(columns=["PlacementStatus"], errors="ignore")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(yte, model.predict_proba(Xte)[:, 1])
        ax.plot(fpr, tpr, lw=2, label=name)
    ax.plot([0, 1], [0, 1], ls="--", c=SLATE, lw=1)
    ax.set_title("ROC Curves on Hold-Out Test Set")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False)
    _save(fig, "roc_curves.png")

    for name, model in models.items():
        key = name.lower().replace(" ", "_")
        if hasattr(model, "coef_"):
            coefs = pd.Series(model.coef_[0], index=Xte.columns).sort_values()
            top = pd.concat([coefs.head(8), coefs.tail(8)])
            fig, ax = plt.subplots(figsize=(6.6, 4.6))
            colors = [GREEN if v > 0 else "#f8b4b4" for v in top.values]
            ax.barh(top.index, top.values, color=colors)
            ax.axvline(0, c=SLATE, lw=1)
            ax.set_title("Logistic Regression — Top Coefficients")
            ax.set_xlabel("Weight (log-odds)")
            _save(fig, "lr_coefs.png")
        if hasattr(model, "feature_importances_"):
            imp = pd.Series(model.feature_importances_, index=Xte.columns).sort_values().tail(15)
            fig, ax = plt.subplots(figsize=(6.6, 4.8))
            ax.barh(imp.index, imp.values, color=TEAL)
            ax.set_title(f"{name} — Feature Importance (top 15)")
            ax.set_xlabel("Importance")
            _save(fig, "feat_importance.png")


def compare_chart(metrics):
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    names = list(metrics)
    for i, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        vals = [metrics[n][metric] for n in names]
        xs = np.arange(len(names)) + (i - 1) * 0.26
        ax.bar(xs, vals, width=0.24, color=[BLUE, TEAL, AMBER][i], label=metric.title())
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Model Performance Comparison")
    ax.legend(frameon=False)
    _save(fig, "model_compare.png")


def cluster_charts():
    df = pd.read_csv("data/placement_data.csv").dropna(subset=["CGPA", "AptitudeTestScore", "CodingTestScore"])
    X = df[["CGPA", "AptitudeTestScore", "CodingTestScore"]].copy()
    Xs = StandardScaler().fit_transform(X)

    ks = range(2, 9)
    inertias = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs).inertia_ for k in ks]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(ks, inertias, "-o", c=BLUE, lw=2)
    ax.set_title("K-Means Elbow — Within-Cluster Inertia")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    _save(fig, "elbow.png")

    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xs)
    sil_scores = []
    for k in range(2, 7):
        kmk = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
        sil_scores.append(silhouette_score(Xs, kmk.labels_))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(list(range(2, 7)), sil_scores, color=TEAL, width=0.6)
    for x, v in zip(range(2, 7), sil_scores):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title("Silhouette Score by Cluster Count")
    ax.set_xlabel("k")
    ax.set_ylabel("Average silhouette")
    _save(fig, "silhouette.png")

    fig = plt.figure(figsize=(7.4, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    for c in range(3):
        mask = km.labels_ == c
        ax.scatter(X.loc[mask, "CGPA"], X.loc[mask, "AptitudeTestScore"],
                   X.loc[mask, "CodingTestScore"], s=9, alpha=0.6, color=PALETTE[c], label=f"Cluster {c+1}")
    ax.set_xlabel("CGPA"); ax.set_ylabel("Aptitude"); ax.set_zlabel("Coding")
    ax.set_title("Student Segments — K-Means (k=3)")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, "kmeans_3d.png")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    db = DBSCAN(eps=0.6, min_samples=25).fit(Xs)
    lab = db.labels_
    out = lab == -1
    ax.scatter(Xs[~out, 0], Xs[~out, 1], s=8, alpha=0.55, c=TEAL, label=f"Clusters ({len(set(lab))-1})")
    ax.scatter(Xs[out, 0], Xs[out, 1], s=22, c=AMBER, marker="x", label=f"Outliers ({out.sum()})")
    ax.set_title("DBSCAN — Density-Based Clustering (Student Profiles)")
    ax.set_xlabel("CGPA (standardised)")
    ax.set_ylabel("Aptitude Test Score (standardised)")
    ax.legend(frameon=False)
    _save(fig, "dbscan.png")


def sigmoid_chart():
    x = np.linspace(-8, 8, 300)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

    axes[0].plot(x, 1 / (1 + np.exp(-x)), c=BLUE, lw=2.4)
    axes[0].axhline(0.5, ls="--", c=SLATE, lw=1)
    axes[0].axvline(0, ls="--", c=SLATE, lw=1)
    axes[0].fill_between(x, 0, 1 / (1 + np.exp(-x)), color=BLUE, alpha=.07)
    axes[0].set_title("Sigmoid σ(z) — probability squashing")
    axes[0].set_xlabel("z = w·x + b")
    axes[0].set_ylabel("P(placed)")

    z = np.linspace(-6, 6, 300)
    axes[1].plot(z, np.log2(1 + np.exp(-z)), c=TEAL, lw=2.4, label="y = 1")
    axes[1].plot(z, np.log2(1 + np.exp(z)), c=AMBER, lw=2.4, label="y = 0")
    axes[1].set_title("Cross-entropy loss — punishes confident mistakes")
    axes[1].set_xlabel("logit z")
    axes[1].set_ylabel("loss")
    axes[1].legend(frameon=False)
    _save(fig, "sigmoid.png")


def regularization_chart():
    t = np.linspace(0, 2 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    # loss contours around the unregularised solution
    cx, cy = 2.0, 1.7
    for r, a in [(0.55, .5), (0.95, .38), (1.35, .28), (1.75, .2)]:
        ax.fill(cx + r * 1.15 * np.cos(t), cy + r * 0.75 * np.sin(t),
                color=TEAL, alpha=a * .35, lw=0)
        ax.plot(cx + r * 1.15 * np.cos(t), cy + r * 0.75 * np.sin(t), c=TEAL, lw=1)

    # constraint regions around the origin
    ax.plot(1.25 * np.cos(t), 1.25 * np.sin(t), c=BLUE, lw=2.4, label="L2 ridge — round ball")
    ax.plot([1.25, 0, -1.25, 0, 1.25], [0, 1.25, 0, -1.25, 0],
            c=AMBER, lw=2.4, label="L1 lasso — diamond")

    ax.scatter([cx], [cy], s=46, c=SLATE, zorder=6, label="least-squares solution")
    ax.scatter([0], [1.25], s=80, c=AMBER, zorder=7, marker="o",
               label="L1 touch → w₁ = 0 (sparse)")
    ax.scatter([1.07, 0.88], [0.86, 1.06], s=80, c=BLUE, zorder=7, marker="o",
               label="L2 touch → both weights kept")
    ax.annotate("corner of the\ndiamond = a zero\nweight", xy=(-1.25, 0), xytext=(-3.0, 1.4),
                arrowprops=dict(arrowstyle="->", color=AMBER), fontsize=9.5, color=AMBER)
    ax.set_title("Why L1 produces sparsity — the geometry of regularisation")
    ax.set_xlabel("w₁")
    ax.set_ylabel("w₂")
    ax.set_xlim(-3.4, 4.0)
    ax.set_ylim(-2.0, 3.2)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _save(fig, "regularization.png")


def scaling_chart():
    df = pd.read_csv("data/placement_data.csv")
    cols = ["CGPA", "AptitudeTestScore"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, c in zip(axes, cols):
        raw = df[c].dropna()
        std = (raw - raw.mean()) / raw.std()
        ax.hist(raw, bins=34, alpha=.75, color=SLATE, label="raw")
        ax.hist(std, bins=34, alpha=.65, color=TEAL, label="standardised")
        ax.set_title(f"{c} — before / after StandardScaler")
        ax.legend(frameon=False)
    _save(fig, "scaling.png")


def missing_chart():
    df = pd.read_csv("data/placement_data.csv")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh(miss.index, miss.values, color=AMBER)
    for i, v in enumerate(miss.values):
        ax.text(v + 60, i, f"{v:,}", va="center", fontsize=9)
    ax.set_title("Missing values by column (median imputation in M1)")
    ax.set_xlabel("Missing cells")
    _save(fig, "missing.png")


def bias_variance_chart():
    from sklearn.tree import DecisionTreeClassifier
    from src.config import PRE_TEST, PRE_TRAIN
    tr, te = pd.read_csv(PRE_TRAIN), pd.read_csv(PRE_TEST)
    Xtr, ytr = tr.drop(columns=["PlacementStatus"]), tr["PlacementStatus"]
    Xte, yte = te.drop(columns=["PlacementStatus"]), te["PlacementStatus"]
    depths = [1, 2, 3, 4, 6, 8, 12, 16, 24, None]
    labels, tr_acc, te_acc = [], [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, min_samples_leaf=25, random_state=42).fit(Xtr, ytr)
        labels.append(str(d) if d else "full")
        tr_acc.append(m.score(Xtr, ytr))
        te_acc.append(m.score(Xte, yte))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(labels, tr_acc, "-o", c=BLUE, label="Training accuracy")
    ax.plot(labels, te_acc, "-o", c=AMBER, label="Test accuracy")
    ax.set_title("Bias–Variance: depth of a single decision tree")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False)
    ax.annotate("underfit", xy=(0, (tr_acc[0] + te_acc[0]) / 2), fontsize=10, c=SLATE)
    ax.annotate("overfit", xy=(len(labels) - 1, tr_acc[-1] - .03), fontsize=10, c=SLATE)
    _save(fig, "bias_variance.png")


def dim_reduction_chart():
    df = pd.read_csv("data/placement_data.csv")
    cols = ["SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
            "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
            "CGPA", "AttendancePercent", "AptitudeTestScore",
            "CodingTestScore", "MockInterviewScore", "SoftSkillsRating"]
    X = StandardScaler().fit_transform(df[cols].dropna())
    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    axes[0].bar(range(1, len(cum) + 1), pca.explained_variance_ratio_, color=TEAL, label="per component")
    axes[0].plot(range(1, len(cum) + 1), cum, "-o", c=BLUE, lw=2, label="cumulative")
    axes[0].axhline(0.9, ls="--", c=AMBER, lw=1.4, label="90% variance")
    axes[0].set_title("PCA — explained variance")
    axes[0].set_xlabel("Principal component")
    axes[0].legend(frameon=False, fontsize=9)

    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(X[:400], method="ward")
    dendrogram(Z, truncate_mode="lastp", p=12, ax=axes[1],
               leaf_rotation=90, leaf_font_size=8, color_threshold=25)
    axes[1].set_title("Hierarchical clustering — dendrogram (400 students)")
    axes[1].set_xlabel("Clustered students")
    axes[1].set_ylabel("Ward distance")
    _save(fig, "dim_reduction.png")


def anomaly_chart():
    df = pd.read_csv("data/placement_data.csv")
    X = df[["CGPA", "AptitudeTestScore"]].dropna()
    iso = IsolationForest(contamination=0.03, random_state=42).fit_predict(X)
    out = iso == -1
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(X.loc[~out, "CGPA"], X.loc[~out, "AptitudeTestScore"],
               s=9, alpha=.5, c=TEAL, label="normal profiles")
    ax.scatter(X.loc[out, "CGPA"], X.loc[out, "AptitudeTestScore"],
               s=26, c=AMBER, marker="x", label=f"anomalies ({out.sum()})")
    ax.set_title("Isolation Forest — anomaly detection (no labels used)")
    ax.set_xlabel("CGPA")
    ax.set_ylabel("Aptitude Test Score")
    ax.legend(frameon=False)
    _save(fig, "anomaly.png")


def confusion_chart(metrics, best_name):
    m = metrics[best_name]["confusion"]
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("cm", ["#ffffff", "#bfdbfe", BLUE]))
    ax.grid(False)
    ax.set_xticks([0, 1], ["Not Placed", "Placed"])
    ax.set_yticks([0, 1], ["Not Placed", "Placed"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=15, color="white" if cm[i, j] > cm.max() / 2 else "#0f172a")
    _save(fig, "confusion.png")