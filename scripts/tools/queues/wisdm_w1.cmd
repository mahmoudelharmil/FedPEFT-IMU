@echo off
rem WISDM queue 1 - leakage-free protocol B, the main grid (one run at a time):
rem   HALO self-supervised pretraining on the pool users, centralized references,
rem   federated methods per encoder; UniMTS (the slowest) last.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\03_pretrain_halo.py --profile wisdm >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm --encoders limu_bert,lstm,ssl_wearables,halo >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v1_head_only,v2_lora_naive,v3_lora_exact,v4_hetero_rank,v5_proposed,ditto_ft >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders lstm >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders halo >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders ssl_wearables >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm --encoders unimts >> logs\wisdm\w1.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders unimts >> logs\wisdm\w1.out 2>&1
echo w1 done >> logs\wisdm\w1.out
