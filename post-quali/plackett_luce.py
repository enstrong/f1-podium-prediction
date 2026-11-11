"""Post-quali Plackett-Luce model: pre-weekend features + grid + grid x circuit pole-win rate.

Run:   python post-quali/plackett_luce.py
       python post-quali/plackett_luce.py --set circuit_shrink=3 --show-loss

The grid baseline is the bar: "finish where you start" is right about the winner more often than not.
Knobs: PARAMS in core/pipeline.py. Sweep them with `python -m core.sweep --mode post`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baselines import grid_order, quali_order  # noqa: E402

from core.pipeline import cli  # noqa: E402

if __name__ == "__main__":
    cli("post", {"grid": grid_order, "quali": quali_order}, Path(__file__).parent)
