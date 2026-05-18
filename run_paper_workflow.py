from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete paper IED-GCN workflow.")
    parser.add_argument("--features", type=Path, default=ROOT / "paper_ied_gcn_inputs" / "features.npy")
    parser.add_argument("--labels", type=Path, default=ROOT / "paper_ied_gcn_inputs" / "labels.npy")
    parser.add_argument("--prepare-script", type=Path, default=ROOT / "prepare_paper_ied_gcn_inputs.py")
    parser.add_argument("--train-script", type=Path, default=ROOT / "run_paper_ied_gcn.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper_ied_gcn_outputs")
    parser.add_argument("--final-dir", type=Path, default=ROOT / "final_outputs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--n-segments", type=int, default=300)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--edge-drop-rate", type=float, default=0.1)
    parser.add_argument("--l1-lambda", type=float, default=1e-5)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", type=str, default="FINAL_150E")
    return parser.parse_args()


def get_python_executable() -> str:
    if DEFAULT_PYTHON.exists():
        return str(DEFAULT_PYTHON)
    return sys.executable


def run_command(command: list[str]) -> None:
    print("\n>>", " ".join(command))
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def package_outputs(source_dir: Path, destination_dir: Path, tag: str) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "report.json": f"{tag}_report.json",
        "training_log.csv": f"{tag}_training_log.csv",
        "ied_gcn_probability_map.npy": f"{tag}_probability_map.npy",
        "ied_gcn_output_5class_jenks.npy": f"{tag}_jenks_5class_map.npy",
        "factor_reference.txt": f"{tag}_factor_reference.txt",
        "factor_reference.png": f"{tag}_factor_reference.png",
        "factor_reference.md": f"{tag}_factor_reference.md",
        "factor_reference.json": f"{tag}_factor_reference.json",
    }

    for source_name, target_name in mapping.items():
        source_path = source_dir / source_name
        if source_path.exists():
            shutil.copy2(source_path, destination_dir / target_name)

    class_map_path = source_dir / "ied_gcn_output_5class_jenks.npy"
    if class_map_path.exists():
        class_map = np.load(class_map_path)
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        image = ax.imshow(class_map, cmap="RdYlGn", vmin=0, vmax=4)
        ax.set_title("Paper IED-GCN 5-Class Susceptibility Map")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
        fig.savefig(destination_dir / "final_map.png", dpi=220)
        plt.close(fig)

    report_path = destination_dir / f"{tag}_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        report.setdefault("outputs", {})["final_map_png"] = str((destination_dir / "final_map.png").resolve())
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


def main() -> int:
    args = parse_args()
    python = get_python_executable()

    run_command([python, str(args.prepare_script)])
    run_command(
        [
            python,
            str(args.train_script),
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
            "--edge-drop-rate",
            str(args.edge_drop_rate),
            "--l1-lambda",
            str(args.l1_lambda),
            "--train-fraction",
            str(args.train_fraction),
            "--seed",
            str(args.seed),
        ]
    )

    run_command([python, str(ROOT / "manage_outputs.py"), "init-view"])
    package_outputs(args.output_dir, args.final_dir, args.tag)
    run_command([python, str(ROOT / "manage_outputs.py"), "init-view"])

    print("\nWorkflow complete.")
    print(f"Final outputs: {args.final_dir}")
    print(f"Live view: {ROOT / 'outputs_view'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())