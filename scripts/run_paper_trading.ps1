# HMATS Paper Trading
# Run this in a separate PowerShell window.

Set-Location -LiteralPath "C:\Users\melod\Downloads\hmats"
& ".\venv\Scripts\Activate.ps1"

Write-Host "=" * 60
Write-Host "HMATS Paper Trading"
Write-Host "Started: $(Get-Date)"
Write-Host "=" * 60

python -X utf8 -u main.py --mode paper

Write-Host "Paper trading exited: $(Get-Date)"
Read-Host "Press Enter to close"
