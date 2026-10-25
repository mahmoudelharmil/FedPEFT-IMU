@echo off
rem Queue 3: published protocol (A) runs, then extra split seeds (1, 2) of the key protocol-B models.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile harbox_fedchar --encoders mlp,limu_bert >> logs\harbox\q3.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox_fedchar --encoders mlp >> logs\harbox\q3.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox_fedchar --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed >> logs\harbox\q3.out 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile harbox --encoders limu_bert,mlp split.seeds=[1,2] >> logs\harbox\q3.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders mlp split.seeds=[1,2] >> logs\harbox\q3.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v3_lora_exact,v5_proposed split.seeds=[1,2] >> logs\harbox\q3.out 2>&1
echo queue3 done >> logs\harbox\q3.out
