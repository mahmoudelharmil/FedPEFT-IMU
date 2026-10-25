@echo off
rem Queue 16 (HARBox): the last runs, one at a time so that at most two processes share the 6 GB GPU
rem (the UniMTS queue runs in parallel). With more, Windows spills GPU memory into system RAM and
rem training becomes >10x slower. Starts when queues 3 and 13 have finished.
rem   1. HALO V5 and ssl-wearables V5 again, storing their new-user models (transductive zero-shot;
rem      ssl-wearables also gets the few-shot BatchNorm fix)
rem   2. seeds 1 and 2 of V5 with LoRA on every LIMU-BERT layer and of Ditto on the pretrained LIMU-BERT
rem   3. the LoRA rank sweep of V3 on LIMU-BERT (ranks 2, 4, 16)
rem   4. Ditto on the pretrained LIMU-BERT under the published protocol A (slowest)
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
:wait
findstr /c:"queue3 done" logs\harbox\q3.out >nul 2>&1 || (timeout /t 60 /nobreak >nul & goto wait)
findstr /c:"queue13 done" logs\harbox\q13.out >nul 2>&1 || (timeout /t 60 /nobreak >nul & goto wait)
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders halo --methods v5_proposed --rerun >> logs\harbox\q16.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders ssl_wearables --methods v5_proposed --rerun >> logs\harbox\q16.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag loraall encoders.limu_bert.lora_scope=all split.seeds=[1,2] >> logs\harbox\q16.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods ditto_ft split.seeds=[1,2] >> logs\harbox\q16.out 2>&1
.venv\Scripts\python.exe scripts\06_run_ablations.py --profile harbox >> logs\harbox\q16.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox_fedchar --encoders limu_bert --methods ditto_ft >> logs\harbox\q16.out 2>&1
echo queue16 done >> logs\harbox\q16.out
