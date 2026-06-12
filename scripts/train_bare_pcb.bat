@echo off
cd /d "%~dp0.."
python src\bare_pcb\train_yolov8.py --data-root data\bare_pcb %*
