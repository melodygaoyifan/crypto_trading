# HMATS DRL Training — Sequential (BTC → ETH → SOL)
# Run this in a PowerShell window. It will survive Claude Code session closure.

Set-Location -LiteralPath "C:\Users\melod\Downloads\hmats"
& ".\venv\Scripts\Activate.ps1"

Write-Host "=" * 60
Write-Host "HMATS DRL Training — Sequential"
Write-Host "Started: $(Get-Date)"
Write-Host "=" * 60

# BTC (2.5M steps x 3 folds)
Write-Host "`n>>> BTC Training Starting..."
python -X utf8 -u train_drl_full.py --asset BTC --folds 3 --no-progress-bar
Write-Host ">>> BTC Training Complete: $(Get-Date)"

# ETH (2.5M steps x 3 folds)
Write-Host "`n>>> ETH Training Starting..."
python -X utf8 -u train_drl_full.py --asset ETH --folds 3 --no-progress-bar
Write-Host ">>> ETH Training Complete: $(Get-Date)"

# SOL (3.0M steps x 3 folds)
Write-Host "`n>>> SOL Training Starting..."
python -X utf8 -u train_drl_full.py --asset SOL --folds 3 --no-progress-bar
Write-Host ">>> SOL Training Complete: $(Get-Date)"

Write-Host "`n" "=" * 60
Write-Host "ALL TRAINING COMPLETE: $(Get-Date)"
Write-Host "=" * 60
Read-Host "Press Enter to close"
