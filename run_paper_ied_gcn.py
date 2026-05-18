from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-only IED-GCN implementation")
    parser.add_argument("--features", type=Path, default=ROOT / "paper_ied_gcn_inputs" / "features.npy")
    parser.add_argument("--labels", type=Path, default=ROOT / "paper_ied_gcn_inputs" / "labels.npy")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_ied_gcn_outputs")
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--edge-drop-rate", type=float, default=0.1)
    parser.add_argument("--l1-lambda", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        "-m",
        "paper_ied_gcn.train",
        "--features",
        str(args.features),
        "--labels",
        str(args.labels),
        "--output-dir",
        str(args.output_dir),
        "--n-segments",
        str(args.n_segments),
        "--compactness",
        str(args.compactness),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--l1-lambda",
        str(args.l1_lambda),
        "--edge-drop-rate",
        str(args.edge_drop_rate),
        "--train-fraction",
        str(args.train_fraction),
        "--seed",
        str(args.seed),
    ]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

