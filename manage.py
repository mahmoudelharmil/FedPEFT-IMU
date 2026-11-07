"""
Pipeline manager of FedPEFT-IMU: plan, run and monitor every experiment from a browser.

    .venv\\Scripts\\python manage.py                  # UI on http://localhost:8765 (opens the browser)
    .venv\\Scripts\\python manage.py --port 9000 --no-browser
    .venv\\Scripts\\python manage.py run-all          # no UI: queue every missing job of the 4 datasets, start
    .venv\\Scripts\\python manage.py run-all --datasets ucihar,wisdm --from-stage ablation
    .venv\\Scripts\\python manage.py status           # what is done, queued and running
    .venv\\Scripts\\python manage.py runner           # run the queue in this terminal (instead of detached)

The jobs run in a detached background runner, one at a time; closing the UI or the
terminal does not stop them.  What "the full process" is (stages, ablations, trials)
is defined in configs/manager.yaml; the training settings are the profiles in
configs/profiles/ (both editable in the UI).
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from fedpeft.manager import state
from fedpeft.manager.plan import STAGES


def cmd_ui(args) -> None:
    from fedpeft.manager.server import serve
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
    serve(args.port)


def cmd_run_all(args) -> None:
    from fedpeft.manager.server import enqueue
    body = {"datasets": args.datasets.split(",") if args.datasets else None, "from_stage": args.from_stage,
            "stages": args.stages.split(",") if args.stages else None, "start": True}
    out = enqueue(body)
    print(f"queued {out['added']} new jobs ({out['requested']} missing); runner: {out['runner']}")


def cmd_status(_args) -> None:
    from fedpeft.manager.server import overview
    o = overview()
    r = o["runner"]
    print(f"runner: {'alive' if r.get('alive') else 'not running'} ({r.get('state', '-')}), "
          f"queue: {o['counts']}, missing work ~{o['missing_min'] / 60:.1f} h")
    for d in o["datasets"]:
        cells = [f"{s['stage']} {s['done']}/{s['total']}" for s in d["stages"]]
        print(f"  {d['name']:<11} " + " · ".join(cells))
    if o["current"]:
        p = o["current"]["progress"]
        print(f"current: {o['current']['item']['id']}  {p['phase']} {p['step']}/{p['total']}")


def cmd_runner(_args) -> None:
    from fedpeft.manager.runner import main
    main()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ui = sub.add_parser("ui", help="start the web UI (default)")
    for p in (ap, ui):
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-browser", action="store_true")
    ra = sub.add_parser("run-all", help="queue every missing job and start the runner")
    ra.add_argument("--datasets", help="comma separated (default: all, in configs/manager.yaml order)")
    ra.add_argument("--from-stage", choices=STAGES, help="skip the stages before this one")
    ra.add_argument("--stages", help="comma separated subset of stages")
    sub.add_parser("status", help="print the progress")
    sub.add_parser("runner", help="run the queue in this terminal")
    a = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    {"run-all": cmd_run_all, "status": cmd_status, "runner": cmd_runner}.get(a.cmd, cmd_ui)(a)
