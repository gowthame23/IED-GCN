"""Run one model on its own: python run_experiments.py paper   (or: ours)

Trains the model for several seeds on two splits and writes results/<model>/:
  run_<split>_s<seed>.json   metrics of each run (seed 0 also saves maps + landslide density)
  summary.json / summary.md  mean +- std over seeds
Splits: 'random' = the paper's random 70/30 split, 'spatial' = whole geographic blocks held out.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (PROJECT, RESULTS_DIR, HyperParams, build_graph_data, classification_metrics, landslide_density,  # noqa: E402
                    load_data, save_map_png, split_nodes, valid_pixels, valid_superpixels)

SPLITS = ("random", "spatial")
METRICS = ["auc", "oa", "precision", "recall", "f1", "mcc", "kappa"]


def run_job(model_name: str, split: str, seed: int, threads: int, epochs: int) -> None:
    import torch

    torch.set_num_threads(threads)
    model = importlib.import_module(f"{model_name}_model")
    out = RESULTS_DIR / model_name
    out.mkdir(parents=True, exist_ok=True)
    hp = HyperParams(epochs=epochs)
    features, labels = load_data()
    g = build_graph_data(features, model.model_features(features), labels, hp)  # superpixels cut from the 16 raw factors
    train_idx, test_idx = split_nodes(g, split, seed, hp.train_fraction)
    print(f"[{model_name}/{split}/s{seed}] superpixels={g.n} train={len(train_idx)} test={len(test_idx)} "
          f"mixed-label superpixels={g.mixed_label_superpixels}", flush=True)

    prob, extras = model.fit_predict(g, train_idx, hp, seed)
    y_test = g.sp_labels[test_idx]
    result = {"model": model_name, "split": split, "seed": seed, "superpixels": g.n,
              "train_nodes": int(len(train_idx)), "test_nodes": int(len(test_idx)),
              "metrics": classification_metrics(y_test, prob[test_idx])}
    if "single_member_prob" in extras:
        result["metrics_single_member"] = classification_metrics(y_test, extras["single_member_prob"][test_idx])
    print(f"[{model_name}/{split}/s{seed}] " + " ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items()), flush=True)

    if seed == 0:  # maps and landslide density from the first run of each split
        valid = valid_pixels(features)
        classes = model.classify(prob, valid_superpixels(g, valid))
        class_map = classes[g.segments].astype(np.int8)
        held = np.zeros(g.n, dtype=bool)
        held[test_idx] = True
        all_pos = np.nonzero(labels == 1)
        test_pos = np.nonzero((labels == 1) & held[g.segments])
        np.save(out / f"probability_map_{split}.npy", prob[g.segments].astype(np.float32))
        np.save(out / f"susceptibility_5class_{split}.npy", class_map)
        result["landslide_density_all"] = landslide_density(class_map, all_pos, valid)
        result["landslide_density_heldout"] = landslide_density(class_map, test_pos, valid)
        save_map_png(out / f"susceptibility_map_{split}.png", class_map, all_pos,
                     f"{model.TITLE}: 5-class susceptibility ({split} split, seed 0)", valid)
    (out / f"run_{split}_s{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def aggregate(model_name: str) -> None:
    model = importlib.import_module(f"{model_name}_model")
    out = RESULTS_DIR / model_name
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("run_*.json"))]
    summary: dict = {"model": model_name, "title": model.TITLE, "splits": {}}
    lines = [f"# {model.TITLE}", "", "Mean ± std over seeds.", "",
             "| Split | Runs | " + " | ".join(m.upper() for m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    extra_lines = []
    for split in SPLITS:
        rs = [r for r in runs if r["split"] == split]
        if not rs:
            continue

        def stats(key: str) -> dict[str, list[float]]:
            return {m: [float(np.mean([r[key][m] for r in rs])), float(np.std([r[key][m] for r in rs]))] for m in METRICS}

        entry = {"runs": len(rs), "metrics": stats("metrics")}
        lines.append(f"| {split} | {len(rs)} | " + " | ".join(f"{entry['metrics'][m][0]:.4f} ± {entry['metrics'][m][1]:.4f}" for m in METRICS) + " |")
        if all("metrics_single_member" in r for r in rs):
            entry["metrics_single_member"] = stats("metrics_single_member")
            extra_lines.append(f"| {split} (first ensemble member only) | {len(rs)} | " + " | ".join(
                f"{entry['metrics_single_member'][m][0]:.4f} ± {entry['metrics_single_member'][m][1]:.4f}" for m in METRICS) + " |")
        seed0 = next((r for r in rs if r["seed"] == 0), None)
        if seed0 and "landslide_density_all" in seed0:
            entry["landslide_density_all"] = seed0["landslide_density_all"]
            entry["landslide_density_heldout"] = seed0["landslide_density_heldout"]
        summary["splits"][split] = entry
    lines += extra_lines
    for split, entry in summary["splits"].items():
        for key, title in (("landslide_density_all", "all landslide samples"), ("landslide_density_heldout", "held-out landslide samples")):
            if key in entry:
                lines += ["", f"### Landslide density, {title} ({split} split, seed 0)", "",
                          "| Class | PC (area share) | PL (landslide share) | LD |", "|---|---|---|---|"]
                lines += [f"| {r['class']} | {r['PC']:.4f} | {r['PL']:.4f} | {r['LD']:.2f} |" for r in entry[key]]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=["paper", "ours"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--parallel", type=int, default=4, help="runs at the same time")
    parser.add_argument("--threads", type=int, default=3, help="torch threads per run")
    parser.add_argument("--epochs", type=int, default=HyperParams.epochs)
    parser.add_argument("--job", action="store_true", help="internal: run a single split/seed")
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.job:
        run_job(args.model, args.split, args.seed, args.threads, args.epochs)
        return 0

    venv_python = PROJECT.parent / ".venv" / "Scripts" / "python.exe"
    python = str(venv_python) if venv_python.exists() else sys.executable
    jobs = [[python, str(Path(__file__).resolve()), args.model, "--job", "--split", split, "--seed", str(seed),
             "--threads", str(args.threads), "--epochs", str(args.epochs)] for seed in range(args.seeds) for split in SPLITS]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        codes = list(pool.map(lambda cmd: subprocess.call(cmd, cwd=str(PROJECT)), jobs))
    if any(codes):
        print(f"{sum(c != 0 for c in codes)} run(s) failed", file=sys.stderr)
        return 1
    aggregate(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
