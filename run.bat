@echo off
REM Skin Cancer Model Training Pipeline - Run Script (Windows)
echo.
echo ============================================================
echo    Skin Cancer Model Training Pipeline
echo ============================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Set PYTHONPATH
set PYTHONPATH=%CD%

REM Run the workflow
echo Running workflow...
echo.
python app\workflow.py %*

echo.
echo Pipeline complete!
echo Check reports\validation_report.html for detailed analysis
echo.
pause
