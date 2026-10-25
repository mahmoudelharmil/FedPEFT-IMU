@echo off
rem Queue 8 (HARBox, LIMU-BERT): V5 whose final personalisation also fine-tunes a private copy of the
rem LoRA adapters (never uploaded), with the default and a 3x longer personalisation budget.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag plora fl.personalize_lora=true >> logs\harbox\q8.out 2>&1
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed --tag plora60 fl.personalize_lora=true fl.final_personalization_steps=60 >> logs\harbox\q8.out 2>&1
echo queue8 done >> logs\harbox\q8.out
