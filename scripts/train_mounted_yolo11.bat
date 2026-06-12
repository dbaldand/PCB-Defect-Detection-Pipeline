@echo off
cd /d "%~dp0.."
python src\mounted_pcb\train_yolo11_4class.py --data mounted_data.yaml %*
