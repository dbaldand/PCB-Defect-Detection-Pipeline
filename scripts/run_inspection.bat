@echo off
cd /d "%~dp0.."
if not exist "src\configs\inspection_config.yaml" (
  echo Copy src\configs\inspection_config.yaml.example to src\configs\inspection_config.yaml first.
  exit /b 1
)
streamlit run src\ui\app.py
