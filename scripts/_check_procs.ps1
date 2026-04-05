$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('streamlit') }
if ($procs) {
    foreach ($p in $procs) {
        Write-Host "PID=$($p.ProcessId) CMD=$($p.CommandLine.Substring(0, [Math]::Min(150, $p.CommandLine.Length)))"
    }
} else {
    Write-Host "NO_STREAMLIT_PROCESSES"
}

Write-Host "---"

$paper = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('main.py') -and $_.CommandLine.Contains('paper') }
if ($paper) {
    foreach ($p in $paper) {
        Write-Host "PID=$($p.ProcessId) CMD=$($p.CommandLine.Substring(0, [Math]::Min(150, $p.CommandLine.Length)))"
    }
} else {
    Write-Host "NO_PAPER_RUN_PROCESSES"
}
