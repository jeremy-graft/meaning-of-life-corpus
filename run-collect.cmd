@echo off
REM Daily YouTube collection. The quota window resets every 24h and is free,
REM so an unused day is simply lost. Alternates the two multilingual seed sets.
cd /d "D:\Claude Code\meaning-of-life-corpus"
set SEEDS=config\seeds_multilingual_2.yaml
if "%date:~4,2%"=="" goto run
set /a DAY=1%date:~0,2% %% 2
if %DAY%==0 set SEEDS=config\seeds_multilingual.yaml
:run
"C:\Users\jerrr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" run python cli.py collect --seeds %SEEDS% >> collect.log 2>&1
