"""Build results/comparison.md from the finished paper-model and our-model runs.

    python compare.py        (after: python run_experiments.py paper ; python run_experiments.py ours)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, SPATIAL_BLOCK_PX, classification_metrics, load_data  # noqa: E402

METRICS = ["auc", "oa", "precision", "recall", "f1", "mcc", "kappa"]
# Reported in the paper's text (Sec. IV-C / IV-D): AUC, OA, Precision, Recall, F1, MCC, Kappa.
PUBLISHED = {"auc": 0.9838, "oa": 0.9838, "precision": 0.9772, "recall": 0.9907, "f1": 0.9839, "mcc": 0.9676, "kappa": 0.9675}


def load(model: str) -> dict:
    path = RESULTS_DIR / model / "summary.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - run: python run_experiments.py {model}")
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(stat: list[float]) -> str:
    return f"{stat[0]:.4f} ± {stat[1]:.4f}"


def metric_row(label: str, split: str, stats: dict) -> str:
    return f"| {label} | {split} | " + " | ".join(fmt(stats[m]) for m in METRICS) + " |"


def pixel_baselines(features: np.ndarray, labels: np.ndarray, split: str, seed: int) -> dict[str, dict[str, float]]:
    """Logistic regression / random forest on the raw 16 factors at labeled pixels, split the same way."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rows, cols = np.nonzero(labels >= 0)
    x, y = features[rows, cols], labels[rows, cols]
    rng = np.random.default_rng(seed)
    if split == "random":
        order = rng.permutation(len(y))
        k = int(len(y) * 0.7)
        tr, te = order[:k], order[k:]
    else:
        block = (rows // SPATIAL_BLOCK_PX) * 1000 + (cols // SPATIAL_BLOCK_PX)
        blocks = np.unique(block)
        held = rng.choice(blocks, max(1, int(round(len(blocks) * 0.3))), replace=False)
        te, tr = np.flatnonzero(np.isin(block, held)), np.flatnonzero(~np.isin(block, held))
    out = {}
    for name, model in (("Logistic regression", LogisticRegression(max_iter=2000)),
                        ("Random forest", RandomForestClassifier(300, n_jobs=-1, random_state=seed))):
        model.fit(x[tr], y[tr])
        out[name] = classification_metrics(y[te], model.predict_proba(x[te])[:, 1])
    return out


def main() -> None:
    paper, ours = load("paper"), load("ours")
    header = "| Model | Split | " + " | ".join(m.upper() for m in METRICS) + " |\n|---|---|" + "---|" * len(METRICS)
    lines = ["# Paper model vs our model", "",
             "Same data (`data/`), same superpixels (np=10), same splits, same hyper-parameters. Mean ± std over seeds.", ""]

    lines += ["## 1. Main comparison", "", header,
              "| *Published in the paper (different data)* | random 70/30 | " + " | ".join(f"{PUBLISHED[m]:.4f}" for m in METRICS) + " |"]
    for split in ("random", "spatial"):
        lines.append(metric_row("Paper model", split, paper["splits"][split]["metrics"]))
        lines.append(metric_row("Our model", split, ours["splits"][split]["metrics"]))

    lines += ["", "## 2. Susceptibility map: landslide density on held-out landslide samples (seed 0)", "",
              "A good map has density rising with class. The paper's reported figure is 97.81% of landslides in "
              "High + Very High, covering 10.25% of the area.", "",
              "| Model | Split | " + " | ".join(f"{c} area% / LD" for c in ["VeL", "Lo", "Mo", "Hi", "VeH"]) + " | Hi+VeH area | Hi+VeH landslides |", "|---|---|---|---|---|---|---|---|---|"]
    for split in ("random", "spatial"):
        for label, res in (("Paper model", paper), ("Our model", ours)):
            ld = res["splits"][split].get("landslide_density_heldout")
            if not ld:
                continue
            top_area = sum(r["PC"] for r in ld[3:]) * 100
            top_hit = sum(r["PL"] for r in ld[3:]) * 100
            lines.append(f"| {label} | {split} | " + " | ".join(f"{r['PC'] * 100:.1f}% / {r['LD']:.2f}" for r in ld) + f" | {top_area:.1f}% | {top_hit:.1f}% |")

    features, labels = load_data()
    seeds = range(paper["splits"]["random"]["runs"])
    base: dict[tuple[str, str], list[dict]] = {}
    for split in ("random", "spatial"):
        for seed in seeds:
            for name, m in pixel_baselines(features, labels, split, seed).items():
                base.setdefault((name, split), []).append(m)
    lines += ["", "## 3. Reference: simple models on the raw factors (not part of the paper)", "",
              "These show how easy the labels are: a plain logistic regression is already close to the top.", "", header]
    for (name, split), items in sorted(base.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        stats = {m: [float(np.mean([i[m] for i in items])), float(np.std([i[m] for i in items]))] for m in METRICS}
        lines.append(metric_row(name, split, stats))

    (RESULTS_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {RESULTS_DIR / 'comparison.md'}")


if __name__ == "__main__":
    main()
