@echo off
rem Queue 2 (HARBox, protocol B): federated ssl-wearables and UniMTS.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders ssl_wearables,unimts >> logs\harbox\q2.out 2>&1
echo queue2 done >> logs\harbox\q2.out
