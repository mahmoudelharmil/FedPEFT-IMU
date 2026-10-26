@echo off
rem Completes the three test-subject draws of the WISDM published protocol (FedAIoT) for the
rem rows that had fewer: ssl-wearables V5 (alpha 0.1: draws 1-2; alpha 0.5: draw 0) and
rem LIMU-BERT FedAvg (alpha 0.5: draws 1-2). Starts after all_final.cmd ("all done").
rem Saved with CRLF line endings.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
set W=logs\wisdm\final2.out
set P=.venv\Scripts\python.exe
:wait
findstr /c:"all done" logs\wisdm\final.out >nul 2>&1 || (timeout /t 60 /nobreak >nul & goto wait)
%P% scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders ssl_wearables --methods v5_proposed >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot --encoders ssl_wearables --methods v5_proposed split.seeds=[1,2] >> %W% 2>&1
%P% scripts\05_run_federated.py --profile wisdm_fedaiot05 --encoders limu_bert --methods fedavg_scratch split.seeds=[1,2] >> %W% 2>&1
echo final2 done >> %W%
