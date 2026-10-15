"""Run only the explicitly requested sixth model after primary completion."""
import json
from pathlib import Path

from medical_copilot.medqa_extension import run_extension

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/evaluation/benchmark-runs"

if __name__ == "__main__":
    result = run_extension(ROOT, BASE / "openweight-medqa-high-20261011", BASE / "ling-sante-medqa-extension-20261011")
    print(json.dumps(result, indent=2))
