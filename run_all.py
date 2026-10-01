"""
Run the complete FedPEFT-IMU pipeline, step by step, for one profile.

    .venv\\Scripts\\python run_all.py --profile standard
    .venv\\Scripts\\python run_all.py --profile quick           # smoke test (minutes)
    .venv\\Scripts\\python run_all.py --profile standard --from 4   # resume at step 4

Steps (each is also a stand-alone script in scripts/):
    1  prepare Capture-24            scripts/01_prepare_data.py
    2  download official weights     scripts/02_download_weights.py
    3  pretrain HALO (SSL)           scripts/03_pretrain_halo.py
    0  benchmark encoders            scripts/00_benchmark_encoders.py
    4  phase 1: centralized          scripts/04_run_centralized.py
    5  phase 2: federated            scripts/05_run_federated.py
    6  ablations                     scripts/06_run_ablations.py
    7  HTML report                   scripts/07_build_report.py
Every step skips work that is already finished, so re-running is cheap.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    (1, "01_prepare_data.py"),
    (2, "02_download_weights.py"),
    (3, "03_pretrain_halo.py"),
    (0, "00_benchmark_encoders.py"),
    (4, "04_run_centralized.py"),
    (5, "05_run_federated.py"),
    (6, "06_run_ablations.py"),
    (7, "07_build_report.py"),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--from", dest="start", type=int, default=1, help="first step number to run (0-7)")
    parser.add_argument("overrides", nargs="*", help="config overrides passed to every step")
    args = parser.parse_args()

    order = [n for n, _ in STEPS]
    start_pos = order.index(args.start) if args.start in order else 0
    for number, script in STEPS[start_pos:]:
        print(f"\n=== step {number}: {script} ===", flush=True)
        cmd = [sys.executable, str(ROOT / "scripts" / script), "--profile", args.profile, *args.overrides]
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"step {number} ({script}) failed with exit code {result.returncode}")
    print("\nAll steps finished. Report:", ROOT / "reports" / f"FedPEFT-IMU_report_{args.profile}.html")
