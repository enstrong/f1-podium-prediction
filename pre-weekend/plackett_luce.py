"""Pre-weekend Plackett-Luce model: linear score on causal form features, ranking loss.

Run:   python pre-weekend/plackett_luce.py
       python pre-weekend/plackett_luce.py --set half_life=10 l2=0.1 top_k=1
       python pre-weekend/plackett_luce.py --seasons 2025 2026 --show-loss

Before every scored race it rebuilds the features and refits the weights on everything
strictly earlier (walk-forward), so the scores are honest out-of-sample numbers.
Knobs: PARAMS in core/pipeline.py. Sweep them with `python -m core.sweep --mode pre`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from baselines import championship, last_race  # noqa: E402

from core.pipeline import cli  # noqa: E402

if __name__ == "__main__":
    cli("pre", {"championship": championship, "last_race": last_race}, Path(__file__).parent)
