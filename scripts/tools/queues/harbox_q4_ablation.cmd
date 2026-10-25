@echo off
rem Queue 4 (HARBox, protocol B, LIMU-BERT): FedPEFT-IMU (V5) and its component ablation.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed,v5_abl_no_local_norm,v5_abl_shared_head,v5_abl_fixed_rank >> logs\harbox\q4.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag noproto model.proto_weight=0.0 >> logs\harbox\q4.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag noaug augment.enabled=false >> logs\harbox\q4.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v0_full_ft --tag noaug augment.enabled=false >> logs\harbox\q4.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag pretrainedinit fl.new_user_local_init=pretrained >> logs\harbox\q4.out 2>&1
echo queue4 done >> logs\harbox\q4.out
