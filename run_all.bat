@echo off
REM One-click run: creates a virtual environment, installs libraries, runs everything.
cd /d "%~dp0"

echo [1/5] Creating virtual environment...
python -m venv .venv || (echo Python 3.11+ is required. & pause & exit /b 1)
call .venv\Scripts\activate.bat

echo [2/5] Installing libraries...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || pip install numpy scipy pandas matplotlib SALib

echo [3/5] Running experiments (about 30-50 seconds)...
python src\experiments.py all || (echo Experiments failed. & pause & exit /b 1)

echo [4/5] Drawing figures...
python src\figures.py || (echo Figures failed. & pause & exit /b 1)

echo [5/5] Writing results summary...
python src\report.py || (echo Report failed. & pause & exit /b 1)

echo.
echo Done. Results: results\RESULTS.md   Figures: figures\
pause
