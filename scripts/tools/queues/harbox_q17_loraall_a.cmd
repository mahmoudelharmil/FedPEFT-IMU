@echo off
rem Queue 17 (HARBox, protocol A): V5 with LoRA on every LIMU-BERT layer under the published protocol,
rem next to Ditto on the pretrained encoder (queue 16).
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox_fedchar --encoders limu_bert --methods v5_proposed --tag loraall encoders.limu_bert.lora_scope=all >> logs\harbox\q17.out 2>&1
echo queue17 done >> logs\harbox\q17.out
