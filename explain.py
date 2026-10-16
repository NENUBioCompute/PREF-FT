"""PREF-FT project entry point: explain.py."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    script = root / "code/pref_ft_shap_project/run_pref_ft_shap.py"
    sys.path.insert(0, str(root / "code"))
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")
