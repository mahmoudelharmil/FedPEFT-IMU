@echo off
rem WISDM queue 2 - the published protocol A (FedAIoT, Dirichlet alpha 0.1 and 0.5), two more
rem user splits of protocol B, and a compact component ablation (LIMU-BERT).
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm_fedaiot --encoders lstm,limu_bert,ssl_wearables >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders lstm --methods fedavg_scratch >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v3_lora_exact,v5_proposed >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders lstm --methods fedavg_scratch >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm --encoders limu_bert,lstm --modes scratch split.seeds=[1,2] >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed,ditto_ft split.seeds=[1,2] >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders lstm split.seeds=[1,2] >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_abl_no_local_norm,v5_abl_shared_head >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag loraqv encoders.limu_bert.lora_scope=qv >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag noaug augment.enabled=false >> logs\wisdm\w2.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag noproto model.proto_weight=0 >> logs\wisdm\w2.out 2>&1
echo w2 done >> logs\wisdm\w2.out
