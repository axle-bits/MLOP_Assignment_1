"""Render the end-to-end architecture diagram for the report.

Run from the repo root:
    ./.venv/Scripts/python docs/figures/architecture/render_diagram.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_W, FIG_H = 16, 9
BOX_FC = "#eef2f7"
BOX_EC = "#3d5a80"
LANE_FC = "#f8f9fb"
LANE_EC = "#c9d3e0"
ACCENT = "#ee6c4d"
TEXT = "#1d2d44"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=200)
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def lane(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=LANE_FC, ec=LANE_EC, lw=1.2))
    ax.text(x + 1.5, y + h - 3.2, title, fontsize=11, fontweight="bold",
            color=TEXT, family="sans-serif")


def box(x, y, w, h, label, accent=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                fc="#ffffff" if not accent else "#fdf0ec",
                                ec=ACCENT if accent else BOX_EC, lw=1.4))
    ax.text(x + w / 2, y + h / 2, label, fontsize=8.5, ha="center",
            va="center", color=TEXT, family="sans-serif")
    return (x, y, w, h)


def arrow(src, dst, label=""):
    sx = src[0] + src[2]
    sy = src[1] + src[3] / 2
    dx = dst[0]
    dy = dst[1] + dst[3] / 2
    ax.add_patch(FancyArrowPatch((sx, sy), (dx, dy), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color=BOX_EC,
                                 connectionstyle="arc3,rad=0.08"))
    if label:
        ax.text((sx + dx) / 2, (sy + dy) / 2 + 1.6, label, fontsize=7.5,
                ha="center", color=TEXT, style="italic")


def arrow_down(src, dst, label=""):
    sx = src[0] + src[2] / 2
    sy = src[1]
    dx = dst[0] + dst[2] / 2
    dy = dst[1] + dst[3]
    ax.add_patch(FancyArrowPatch((sx, sy), (dx, dy), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color=BOX_EC))
    if label:
        ax.text(sx + 1.2, (sy + dy) / 2, label, fontsize=7.5, ha="left",
                color=TEXT, style="italic")


def elbow_down(src, dst, bend_y, label="", label_xy=None):
    """Drop straight down from the right edge of src to bend_y, then run
    into the top of dst. Used to route around the tall K8s lane's title
    text instead of cutting a diagonal straight through it."""
    sx = src[0] + src[2] - 3
    sy = src[1]
    dx = dst[0] + dst[2] / 2
    dy = dst[1] + dst[3]
    ax.plot([sx, sx], [sy, bend_y], color=BOX_EC, lw=1.3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((sx, bend_y), (dx, dy), arrowstyle="-|>",
                                 mutation_scale=14, lw=1.3, color=BOX_EC))
    if label:
        lx, ly = label_xy if label_xy else (sx + 1.2, bend_y + 1.5)
        ax.text(lx, ly, label, fontsize=7.5, ha="left", color=TEXT,
                style="italic")


# Lanes
lane(2, 62, 50, 26, "Data & Features")
lane(56, 62, 48, 26, "Training & Tracking")
lane(108, 62, 50, 26, "Packaging")
lane(2, 32, 76, 26, "CI/CD — GitHub Actions")
lane(82, 2, 76, 56, "Kubernetes (Docker Desktop) — namespace heart-disease")

# Data lane
b_uci = box(5, 66, 13, 10, "UCI\nCleveland\n(303 rows)")
b_prep = box(21, 66, 13, 10, "Preprocess\nbinarize target\ndrop 6 rows")
b_feat = box(37, 66, 12, 10, "Clinical\nfeatures\nRPP / HRR")

# Training lane
b_train = box(59, 66, 20, 10, "GridSearchCV 5-fold\nLR / RF / XGBoost\n(6 runs, seed 785)")
b_mlflow = box(82, 66, 19, 10, "MLflow\nparams / metrics\nROC & CM plots")

# Packaging lane
b_export = box(111, 66, 21, 10, "Best-run export\nargmax test ROC-AUC\n= LR + clinical 0.8817")
b_joblib = box(135, 66, 20, 10, "models/\npipeline.joblib\n+ metadata.json")

# CI lane
b_lint = box(5, 36, 14, 10, "lint\n(ruff)")
b_test = box(22, 36, 14, 10, "test\n(pytest, 39)")
b_smoke = box(39, 36, 16, 10, "train-smoke\n(--quick + artifacts)")
b_docker = box(58, 36, 17, 10, "docker\nbuild + /predict\nsmoke", accent=True)

# K8s lane
# Top row sits lower than the lane title so the diagonal arrows coming down
# from the Packaging/CI-CD lanes have clear whitespace to land in, instead
# of crossing through the "Kubernetes (Docker Desktop)..." title text.
b_image = box(85, 33, 20, 12, "Image\nheart-disease-api:v2\nFastAPI + model", accent=True)
b_pods = box(109, 33, 21, 12, "Deployment\n2 replicas\nprobes /health")
b_svc = box(134, 33, 21, 12, "Service\nLoadBalancer\nlocalhost:80")
b_prom = box(109, 17, 21, 12, "Prometheus\nscrape /metrics\n15s")
b_graf = box(134, 17, 21, 12, "Grafana\nprovisioned\ndashboard :3000")
b_client = box(85, 4, 20, 10, "Client\ncurl / Swagger\n/ Postman")

# Flows
arrow(b_uci, b_prep)
arrow(b_prep, b_feat)
arrow(b_feat, b_train)
arrow(b_train, b_mlflow, "log runs")
arrow(b_mlflow, b_export)
arrow(b_export, b_joblib)
arrow(b_lint, b_test)
arrow(b_test, b_smoke)
arrow(b_smoke, b_docker)
arrow(b_image, b_pods, "run")
arrow(b_pods, b_svc)
arrow(b_prom, b_graf, "datasource")
elbow_down(b_joblib, b_image, 50, "baked in")
arrow_down(b_docker, b_image)
ax.text(76, 39.2, "validates", fontsize=7.5, ha="left", color=TEXT,
        style="italic")
arrow_down(b_pods, b_prom, "/metrics")
ax.add_patch(FancyArrowPatch(
    (b_client[0] + b_client[2] / 2, b_client[1] + b_client[3]),
    (b_svc[0] + 4, b_svc[1]), arrowstyle="<|-|>", mutation_scale=14,
    lw=1.3, color=ACCENT, connectionstyle="arc3,rad=-0.25"))
ax.text(112, 12, "POST /predict → prediction + probability",
        fontsize=7.5, color=TEXT, style="italic")

ax.set_title("Heart Disease Risk API — end-to-end MLOps architecture",
             fontsize=14, fontweight="bold", color=TEXT, pad=14)

out = Path(__file__).parent / "architecture.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
