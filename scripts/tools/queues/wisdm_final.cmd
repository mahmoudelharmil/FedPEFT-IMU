@echo off
rem WISDM - all remaining runs, strictly ONE at a time (two concurrent jobs paged the 16 GB host:
rem UniMTS on 200 frames alone holds ~7 GB). Finished runs are skipped automatically.
rem   1. component ablation (LIMU-BERT, protocol B)
rem   2. published protocol A (FedAIoT schedule, 400 rounds), three draws of the 6 test subjects
rem   3. UniMTS on protocol B (slowest, last)
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
set L=logs\wisdm\final.out
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_abl_shared_head >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag loraqv encoders.limu_bert.lora_scope=qv >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag noaug augment.enabled=false >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders limu_bert --methods v5_proposed --tag noproto model.proto_weight=0 >> %L% 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm_fedaiot --encoders lstm,limu_bert,ssl_wearables split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders lstm --methods fedavg_scratch split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders lstm --methods fedavg_scratch split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders ssl_wearables --methods v0_full_ft,v3_lora_exact,v5_proposed >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders ssl_wearables --methods v0_full_ft >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders ssl_wearables --methods v0_full_ft,v3_lora_exact split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders ssl_wearables,limu_bert --methods v0_full_ft,v5_proposed split.seeds=[1,2] >> %L% 2>&1
.venv\Scripts\python.exe scripts\04_run_centralized.py --profile wisdm --encoders unimts >> %L% 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile wisdm --encoders unimts >> %L% 2>&1
echo final done >> %L%
