@echo off
rem Queue 6 (HARBox, LIMU-BERT): dual-head V5 (federated global classifier for new users).
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag dualhead model.global_head=true >> logs\harbox\q6.out 2>&1
echo queue6 done >> logs\harbox\q6.out
