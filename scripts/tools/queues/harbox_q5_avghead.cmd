@echo off
rem Queue 5 (HARBox, LIMU-BERT): V5 with the clients' average classifier as zero-shot rule for new users.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag avghead fl.new_user_head=average >> logs\harbox\q5.out 2>&1
echo queue5 done >> logs\harbox\q5.out
