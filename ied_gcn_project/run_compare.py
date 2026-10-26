"""Click Run (after BOTH models have finished): writes results/comparison.md."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
venv_python = HERE.parent / ".venv" / "Scripts" / "python.exe"
python = str(venv_python) if venv_python.exists() else sys.executable
raise SystemExit(subprocess.call([python, str(HERE / "compare.py"), *sys.argv[1:]], cwd=str(HERE)))
