Start-Sleep -Seconds 5
try {
    $r = Invoke-WebRequest -Uri 'http://localhost:8501/_stcore/health' -UseBasicParsing -TimeoutSec 5
    Write-Host "Status: $($r.StatusCode) $($r.Content)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
}
