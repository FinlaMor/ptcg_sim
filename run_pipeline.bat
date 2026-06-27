@echo off
title PTCG Simulation Optimization Pipeline
cls

echo ===================================================
echo [1/4] BACKING UP EXISTING LIGHTGBM MODEL...
echo ===================================================
if exist rotom_computer.txt (
    if not exist models_backup mkdir models_backup
    set cur_date=%date:~10,4%%date:~4,2%%date:~7,2%
    set cur_time=%time:~0,2%%time:~3,2%%time:~5,2%
    set cur_time=%cur_time: =0%
    set timestamp=%cur_date%_%cur_time%
    copy rotom_computer.txt models_backup\rotom_computer_backup_%timestamp%.txt >nul
    echo [SUCCESS] Backup saved: models_backup\rotom_computer_backup_%timestamp%.txt
) else (
    echo [NOTICE] No existing 'rotom_computer.txt' found to backup. Proceeding...
)

echo.
echo ===================================================
echo [2/4] RUNNING LOGS PARSER...
echo ===================================================
python parse_logs.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Log parsing failed! Stopping pipeline.
    pause
    exit /b %errorlevel%
)

echo.
echo ===================================================
echo [3/4] STARTING LIGHTGBM MODEL TRAINING...
echo ===================================================
python train_model.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Model training failed! Stopping pipeline.
    pause
    exit /b %errorlevel%
)

echo.
echo ===================================================
echo [4/4] CLEANING UP INTERMEDIATE GAME LOG FILES...
echo ===================================================
echo Purging temporary .txt simulation logs from games\selfplay\...
del /q games\selfplay\*.txt >nul 2>&1
echo [SUCCESS] Log folder cleared! Ready for your next data generation run.

echo.
echo ===================================================
echo PIPELINE COMPLETE: rotom_computer.txt has been updated!
echo ===================================================
pause
