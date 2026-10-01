"""
Step 1 - download Capture-24 and convert it to 10-second windows.

    python scripts/01_prepare_data.py

Creates data/raw/capture24.zip (~6.5 GB), data/raw/capture24/ (CSV files) and
data/processed/P###.npz + data/processed/meta.json.  Safe to re-run: finished
participants are skipped and an interrupted download resumes.
"""

from _common import setup

from fedpeft.data.capture24 import prepare_capture24

if __name__ == "__main__":
    cfg, device, log, _ = setup("01_prepare_data", __doc__)
    meta = prepare_capture24(cfg)
    log.info(f"{len(meta['participants'])} participants ready; classes ({cfg.data.label_scheme}): "
             f"{meta['schemes'][cfg.data.label_scheme]}")
