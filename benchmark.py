"""PREF-FT project entry point: benchmark.py."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    script = root / "code/train_external_test_binary_metrics.py"
    sys.path.insert(0, str(root / "code"))
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
