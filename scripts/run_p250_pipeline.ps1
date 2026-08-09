# [P250] End-to-end DS pipeline run: feature lab -> enriched EDA ->
# full-zoo per-cell selection -> assembly. Sequential; stops on failure.
$py = "C:\Users\melod\Downloads\hmats\venv\Scripts\python.exe"
Set-Location "C:\Users\melod\Downloads\hmats"
& $py -X utf8 -u training/feature_lab.py
if (-not $?) { Write-Output "FEATURE LAB FAILED"; exit 1 }
& $py -X utf8 -u training/regime_model_lab.py --stage eda --engineered
if (-not $?) { Write-Output "EDA FAILED"; exit 1 }
& $py -X utf8 -u training/regime_model_lab.py --stage select --engineered --tag p250_full
if (-not $?) { Write-Output "SELECT FAILED"; exit 1 }
& $py -X utf8 -u training/regime_model_lab.py --stage assemble --engineered --tag p250_full
if (-not $?) { Write-Output "ASSEMBLE FAILED"; exit 1 }
Write-Output "P250 PIPELINE COMPLETE"
