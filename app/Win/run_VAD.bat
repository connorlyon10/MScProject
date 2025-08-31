@echo off
REM Move to the project root
cd /d "%~dp0\..\.."

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Run Streamlit
streamlit run src\app_VAD.py
