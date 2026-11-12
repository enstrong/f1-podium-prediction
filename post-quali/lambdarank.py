"""Model 2 (post mode): LightGBM LambdaRank on the same features as model 1, scored on the same races.

Run:   python post-quali/lambdarank.py
       python post-quali/lambdarank.py --set num_leaves=15 n_trees=400
       python -m core.sweep --mode post --base model='"lgbm"' --only n_trees num_leaves

Model 1 (plackett_luce) is included as a competitor, so the paired difference vs it is printed.
Model 1's own script and predictions are untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baselines import grid_order, quali_order  # noqa: E402

from core.pipeline import cli, make_predictor  # noqa: E402

if __name__ == "__main__":
    cli("post", {"plackett_luce (model 1)": make_predictor({}, "post"), "grid": grid_order, "quali": quali_order},
        Path(__file__).parent, defaults={"model": "lgbm"})
