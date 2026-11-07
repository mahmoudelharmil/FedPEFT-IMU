"""
Pipeline manager: plan, run and monitor the whole FedPEFT-IMU process from a browser.

    python manage.py                 # start the UI on http://localhost:8765
    python manage.py run-all         # queue every missing job of the 4 datasets and run it (no UI)

Modules
    plan.py      every job of the full process (configs/manager.yaml + the profiles)
    results.py   one-line summaries of the result files (scores, run times)
    state.py     the persistent queue and the runner's control flags (logs/manager/)
    runner.py    the background worker: runs the queue one job at a time
    progress.py  reads a job's log: phase, round / epoch, learning curve
    profiles.py  reads and edits the training settings (profiles, dataset files)
    server.py    the HTTP API and the single-page UI in static/
"""
