@echo off
title PTCG Self-Play RL Looping Pipeline

:: Keep this initial CLS so the terminal starts clean at Cycle 1
cls

set GAMES_PER_BATCH=300
set GENERATION_CYCLE=1

:PIPELINE_LOOP
:: --- FIXED: Wiped out the 'cls' command that was erasing old cycles ---

echo.
echo #################################################################
echo   PTCG LEARNING HORIZON - PROCESSING GENERATION CYCLE #%GENERATION_CYCLE%
echo #################################################################
echo [Cycle Timestamp: %date% %time%]
echo.

echo Running %GAMES_PER_BATCH% simulations silently in the background...
python main.py %GAMES_PER_BATCH%
if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR] Game simulation loop crashed!
    pause
    exit /b %errorlevel%
)

if exist rotom_computer.txt (
    if not exist models_backup mkdir models_backup
    set cur_date=%date:~10,4%%date:~4,2%%date:~7,2%
    set cur_time=%time:~0,2%%time:~3,2%%time:~5,2%
    set cur_time=%cur_time: =0%
    set timestamp=%cur_date%_%cur_time%
    copy rotom_computer.txt models_backup\rotom_computer_cycle%GENERATION_CYCLE%_%timestamp%.txt >nul
)

python parse_logs.py
if %errorlevel% neq 0 (
    echo [ERROR] Log compilation failed!
    pause
    exit /b %errorlevel%
)

echo.
echo =================================================================
echo                    VALIDATION GATE ENGINE                       
echo =================================================================
python train_model.py
if %errorlevel% equ 1 (
    echo [ERROR] Model optimization engine hit a critical fault!
    pause
    exit /b %errorlevel%
)
echo =================================================================
echo.

del /q games\selfplay\*.txt >nul 2>&1

echo [CYCLE #%GENERATION_CYCLE% COMPLETE] Resetting loop in 3 seconds...
set /a GENERATION_CYCLE=%GENERATION_CYCLE%+1
timeout /t 3 >nul
goto PIPELINE_LOOP
