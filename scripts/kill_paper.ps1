Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    if ($_.CommandLine -like '*main.py*paper*') {
        Stop-Process -Id $_.ProcessId -Force
        Write-Host "Killed PID $($_.ProcessId): $($_.CommandLine)"
    } else {
        Write-Host "Kept PID $($_.ProcessId): $($_.CommandLine.Substring(0, [Math]::Min(80, $_.CommandLine.Length)))"
    }
}
