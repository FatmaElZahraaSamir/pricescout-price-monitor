@echo off
REM One-click setup and launch on Windows
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing requirements (first run takes a few minutes)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
echo Starting the dashboard at http://localhost:8501 ...
streamlit run app.py
pause
