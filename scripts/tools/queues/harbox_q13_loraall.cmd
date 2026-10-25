@echo off
rem Queue 13 (HARBox, LIMU-BERT): LoRA on every linear layer of the Transformer block instead of
rem W_q, W_v only (V5 and the shared-classifier control V3).
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders limu_bert --methods v5_proposed,v3_lora_exact --tag loraall encoders.limu_bert.lora_scope=all >> logs\harbox\q13.out 2>&1
echo queue13 done >> logs\harbox\q13.out
