@echo off
rem Queue 9 (HARBox, LIMU-BERT): Ditto on the pretrained encoder, protocol B and protocol A.
rem Starts when queue 8 has finished, so at most three runs share the GPU.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
:wait
findstr /c:"queue8 done" logs\harbox\q8.out >nul 2>&1 || (timeout /t 60 /nobreak >nul & goto wait)
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods ditto_ft >> logs\harbox\q9.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox_fedchar --encoders limu_bert --methods ditto_ft >> logs\harbox\q9.out 2>&1
echo queue9 done >> logs\harbox\q9.out
