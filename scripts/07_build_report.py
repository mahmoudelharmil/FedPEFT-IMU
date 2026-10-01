"""
Step 7 - build the interactive HTML report from all result files of a profile.

    python scripts/07_build_report.py [--profile standard]

Output: reports/FedPEFT-IMU_report_<profile>.html  (self-contained, open in any browser)
        reports/report_data_<profile>.json          (every number shown in the report)
"""

from _common import setup

from fedpeft.report.build import build_report

if __name__ == "__main__":
    cfg, device, log, _ = setup("07_build_report", __doc__)
    path = build_report(cfg)
    log.info(f"Open {path} in a browser.")
