@echo off
REM Patient transcript harvesting. Safe to run several times a day.
REM Takes a small bite, stops itself if YouTube starts refusing, resumes next time.
cd /d "D:\Claude Code\meaning-of-life-corpus"
"C:\Users\jerrr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe" run python cli.py harvest-transcripts --limit 120 --delay 9 >> harvest.log 2>&1
