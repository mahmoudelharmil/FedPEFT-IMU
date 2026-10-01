"""
FedPEFT-IMU
===========

Parameter-efficient federated adaptation of pretrained inertial (IMU) encoders
for human activity recognition.

Package layout (read top to bottom, it follows the data flow):

    fedpeft.config        - load YAML config + profile + CLI overrides
    fedpeft.utils         - small helpers (seeding, timing, JSON, downloads)
    fedpeft.data          - Capture-24 preparation, user/time splits, window store
    fedpeft.models        - the 8 model components of the PDF + the 4 encoders
    fedpeft.training      - metrics, local training loop, centralized runner, HALO SSL
    fedpeft.federated     - FL simulator: methods V0-V5, clients, server, aggregation
    fedpeft.report        - builds the interactive HTML report from result JSONs
"""

__version__ = "1.0.0"
