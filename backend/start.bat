@echo off
:: WB Transport Optimizer — Backend Starter
:: Run from repo root or backend/ directory.

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run: python -m venv .venv ^& pip install -r requirements.txt
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo.
echo  WB Transport Optimizer — Backend
echo  API docs:  http://localhost:8000/docs
echo  Health:    http://localhost:8000/health
echo.

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
