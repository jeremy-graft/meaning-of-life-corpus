@echo off
REM Daily YouTube collection. The quota window resets every 24h and is free,
REM so an unused day is simply lost.
REM
REM Rotates across all three seed files by day-of-month modulo 3. Wave 1 is the
REM eight original languages, wave 2 the ten later ones, wave 3 the six seams
REM wave 2 never had. Within a file the collector also shuffles its own seed
REM order each run, so nothing is systematically starved.
cd /d "D:\Claude Code\meaning-of-life-corpus"

for /f "tokens=2 delims==" %%d in ('wmic os get localdatetime /value 2^>nul ^| find "="') do set DT=%%d
set DAY=%DT:~6,2%
set /a PICK=(1%DAY% - 100) %% 3

if %PICK%==0 set SEEDS=config\seeds_multilingual.yaml
if %PICK%==1 set SEEDS=config\seeds_multilingual_2.yaml
if %PICK%==2 set SEEDS=config\seeds_multilingual_3.yaml
if "%SEEDS%"=="" set SEEDS=config\seeds_multilingual_3.yaml

echo [%DATE% %TIME%] using %SEEDS% >> collect.log
"C:\Users\jerrr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" run python cli.py collect --seeds %SEEDS% >> collect.log 2>&1
