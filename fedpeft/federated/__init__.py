"""
Federated learning simulator (phase 2).

    methods.py      what each method trains / sends / keeps (FedAvg scratch, V0-V5)
    aggregation.py  server rules: FedAvg, LoRA naive / exact (FedEx) / exact-SVD + hetero ranks, prototypes
    server.py       global state + streaming aggregation
    comm.py         byte accounting of every upload and download
    simulation.py   the round loop, evaluation, KPIs and the result file

The simulator is a transparent, single-process re-creation of a Flower
simulation: every "message" is an explicit dict of tensors whose fp32 size is
counted, so uplink / downlink numbers are measured, not estimated.
"""
