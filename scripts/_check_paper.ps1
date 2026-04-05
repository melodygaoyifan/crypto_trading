$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and $_.CommandLine.Contains('main.py') -and $_.CommandLine.Contains('paper') }
if ($procs) {
    foreach ($p in $procs) {
        Write-Host "PID=$($p.ProcessId) ParentPID=$($p.ParentProcessId)"
    }
} else {
    Write-Host "NO_PAPER_PROCESSES"
}
