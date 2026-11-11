"""One-click pipeline: pull data, run tests, score both baseline sets (walk-forward, no holdout).
Output also saved to reports/run_all.log."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
STEPS = [
    [sys.executable, "-m", "core.data", "--refresh"],
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "pre-weekend/baselines.py"],
    [sys.executable, "post-quali/baselines.py"],
]

log_dir = ROOT / "reports"
log_dir.mkdir(exist_ok=True)
with open(log_dir / "run_all.log", "w") as log:
    for cmd in STEPS:
        print(f"\n$ {' '.join(cmd[1:])}", flush=True)
        log.write(f"\n$ {' '.join(cmd[1:])}\n")
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out = p.stdout + p.stderr
        print(out, flush=True)
        log.write(out)
        log.flush()
        if p.returncode != 0:
            print(f"FAILED: {' '.join(cmd[1:])}")
            sys.exit(1)
print("\nALL DONE")
