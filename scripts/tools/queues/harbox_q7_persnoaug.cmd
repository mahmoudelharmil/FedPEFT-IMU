@echo off
rem Queue 7 (HARBox, LIMU-BERT): V5 whose final personalisation uses un-augmented windows.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag persnoaug fl.personalize_augment=false >> logs\harbox\q7.out 2>&1
echo queue7 done >> logs\harbox\q7.out
