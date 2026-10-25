@echo off
rem Queue 10 (HARBox): V5 re-runs of the two BatchNorm encoders after the few-shot fix (BatchNorm
rem statistics stay frozen during few-shot adaptation of a new user). ssl-wearables first; then,
rem when queue 2 reaches UniMTS V5 (with the old code), it is stopped and UniMTS V5 runs here.
cd /d E:\Masters\Thesis2\idea1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders ssl_wearables --methods v5_proposed >> logs\harbox\q10.out 2>&1
:wait
findstr /c:"fl.unimts.v5_proposed |" logs\harbox\q2.out >nul 2>&1 || (timeout /t 60 /nobreak >nul & goto wait)
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*05_run_federated.py --profile harbox --encoders ssl_wearables,unimts*' } | Invoke-CimMethod -MethodName Terminate | Out-Null"
echo stopped queue 2 before UniMTS V5 >> logs\harbox\q10.out
.venv\Scripts\python.exe scripts\05_run_federated.py --profile harbox --encoders unimts --methods v5_proposed >> logs\harbox\q10.out 2>&1
echo queue10 done >> logs\harbox\q10.out
