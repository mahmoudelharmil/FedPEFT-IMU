@echo off
rem Queue 11 (HARBox, seed 0): re-runs with the final code, which also stores every new-user model
rem (results/harbox/models) so the held-out protocols can be re-evaluated without re-training
rem (scripts/tools/reeval_heldout.py, e.g. transductive zero-shot). ssl-wearables V5 also gets the
rem few-shot BatchNorm fix.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed,v0_full_ft,fedavg_scratch,v3_lora_exact --rerun >> logs\harbox\q11.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders ssl_wearables --methods v5_proposed --rerun >> logs\harbox\q11.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders mlp --methods fedavg_scratch --rerun >> logs\harbox\q11.out 2>&1
echo queue11 done >> logs\harbox\q11.out
