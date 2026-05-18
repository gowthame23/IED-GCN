@echo off
REM Paper IED-GCN One-Click Workflow Runner
REM Double-click this file to run the complete pipeline and generate all outputs

setlocal enabledelayedexpansion

cd /d "%~dp0"

echo.
echo ================================================================================
echo  PAPER IED-GCN ONE-CLICK WORKFLOW
echo ================================================================================
echo.
echo Running complete paper workflow...
echo  - Prepare inputs
echo  - Train IED-GCN (150 epochs)
echo  - Package outputs
echo  - Refresh live view
echo.

.venv\Scripts\python.exe run_paper_workflow.py --epochs 150

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo  ERROR: Workflow failed!
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo  WORKFLOW COMPLETE
echo ================================================================================
echo.
echo Final outputs are in: final_outputs\
echo Live view is in:      outputs_view\
echo.
echo Opening output folder...
echo.

start explorer.exe "%cd%\final_outputs"

pause
