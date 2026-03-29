@echo off
chcp 65001 >nul
python main.py 2>&1 | tee output\main_log.txt
pause
