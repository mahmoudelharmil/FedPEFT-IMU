@echo off
rem Remaining runs of WISDM and UCI HAR, strictly ONE job at a time (16 GB host). Finished runs are
rem skipped automatically, so this queue can be restarted at any time.
rem   1. UCI HAR, leakage-free protocol B: HALO SSL, centralized, federated (all but UniMTS)
rem   2. UCI HAR, published protocol A (FedMultimodal, alpha 0.1 and 5.0)
rem   3. WISDM, published protocol A (FedAIoT): draw 0 first, then draws 1-2
rem   4. extra user splits (UCI HAR)
rem   5. UniMTS (slowest): UCI HAR, then WISDM
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
set U=logs\ucihar\queue.out
set W=logs\wisdm\final.out
set P=.venv\Scripts\python.exe
%P% scripts\03_pretrain_halo.py --profile ucihar >> %U% 2>&1
%P% scripts\04_run_centralized.py --profile ucihar --encoders limu_bert,convgru,ssl_wearables,halo >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v1_head_only,v2_lora_naive,v3_lora_exact,v4_hetero_rank,v5_proposed,ditto_ft >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders convgru >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders ssl_wearables >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders halo >> %U% 2>&1
%P% scripts\04_run_centralized.py --profile ucihar_fedmm --encoders convgru,limu_bert,ssl_wearables >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar_fedmm --encoders convgru --methods fedavg_scratch >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar_fedmm5 --encoders convgru --methods fedavg_scratch >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar_fedmm --encoders limu_bert,ssl_wearables --methods fedavg_scratch,v0_full_ft,v5_proposed >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar_fedmm5 --encoders limu_bert,ssl_wearables --methods fedavg_scratch,v0_full_ft,v5_proposed >> %U% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders ssl_wearables --methods v0_full_ft,v3_lora_exact,v5_proposed >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders ssl_wearables --methods v0_full_ft >> %W% 2>&1
%P% scripts\04_run_centralized.py --profile wisdm_fedaiot --encoders lstm,limu_bert,ssl_wearables split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders lstm --methods fedavg_scratch split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders lstm --methods fedavg_scratch split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders ssl_wearables --methods v0_full_ft,v3_lora_exact split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders ssl_wearables,limu_bert --methods v0_full_ft,v5_proposed split.seeds=[1,2] >> %W% 2>&1
%P% scripts\04_run_centralized.py --profile ucihar --encoders limu_bert,convgru --modes scratch split.seeds=[1,2] >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders limu_bert --methods fedavg_scratch,v0_full_ft,v5_proposed,ditto_ft split.seeds=[1,2] >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders convgru split.seeds=[1,2] >> %U% 2>&1
%P% scripts\04_run_centralized.py --profile ucihar --encoders unimts >> %U% 2>&1
%P% scripts\05_run_federated.py --profile ucihar --encoders unimts >> %U% 2>&1
%P% scripts\04_run_centralized.py --profile wisdm --encoders unimts >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm --encoders unimts >> %W% 2>&1
echo all done >> %U%
echo all done >> %W%
