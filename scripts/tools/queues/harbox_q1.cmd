@echo off
rem Queue 1 (HARBox, protocol B): centralized phase for every encoder, then federated HALO and MLP.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile harbox >> logs\harbox\q1.out 2>&1 && ^
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders halo,mlp >> logs\harbox\q1.out 2>&1
echo queue1 done >> logs\harbox\q1.out
