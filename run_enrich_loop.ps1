# Self-healing enrichment loop. Run by the "MOL-Enrich" scheduled task so it
# survives Claude-session teardowns and reboots. Resumes from the DB each pass
# (idempotent), exits 0 when the backlog is empty.
$uv = "C:\Users\jerrr\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe"
$d  = "D:\Claude Code\meaning-of-life-corpus"
$log = Join-Path $d "enrich_loop.log"
while ($true) {
  "$(Get-Date -Format o)  pass starting" | Add-Content $log
  & $uv run --directory $d python cli.py enrich --local --prefilter *>> $log
  if ($LASTEXITCODE -eq 0) { "$(Get-Date -Format o)  BACKLOG EMPTY - done" | Add-Content $log; break }
  "$(Get-Date -Format o)  stopped (exit $LASTEXITCODE); resuming in 15s" | Add-Content $log
  Start-Sleep -Seconds 15
}
