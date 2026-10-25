"""
Step 1 - download the dataset and convert it to fixed-size windows.

    python scripts/01_prepare_data.py                      # Capture-24 (default)
    python scripts/01_prepare_data.py --profile harbox     # HARBox
    python scripts/01_prepare_data.py --profile wisdm      # WISDM-2019 (smartwatch)
    python scripts/01_prepare_data.py --profile ucihar     # UCI HAR (waist smartphone)

Capture-24: data/raw/capture24.zip (~6.5 GB) -> data/processed/P###.npz (10-s windows)
HARBox:     data/raw/harbox/ (~175 MB)       -> data/processed_harbox/u###.npz (2-s windows)
WISDM-2019: data/raw/wisdm/ (~310 MB)        -> data/processed_wisdm_watch/s####.npz (10-s windows)
Both write meta.json next to the window files.  Safe to re-run: finished
participants are skipped and an interrupted download resumes.
"""

from _common import setup

if __name__ == "__main__":
    cfg, device, log, _ = setup("01_prepare_data", __doc__)
    if cfg.data.dataset == "harbox":
        from fedpeft.data.harbox import prepare_harbox as prepare
    elif cfg.data.dataset == "wisdm":
        from fedpeft.data.wisdm import prepare_wisdm as prepare
    elif cfg.data.dataset == "ucihar":
        from fedpeft.data.ucihar import prepare_ucihar as prepare
    else:
        from fedpeft.data.capture24 import prepare_capture24 as prepare
    meta = prepare(cfg)
    log.info(f"{len(meta['participants'])} participants ready; classes ({cfg.data.label_scheme}): "
             f"{meta['schemes'][cfg.data.label_scheme]}")
