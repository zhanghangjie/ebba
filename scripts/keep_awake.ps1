# Keep Windows awake while a long full-profile training run is in progress.
#
#   powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\keep_awake.ps1 -AllowDisplayOff
#
# Press Ctrl+C to stop.  This prevents system sleep; it does not protect the
# Docker/CUDA process if you close the laptop lid or force shutdown.  The
# reliable recovery path is always checkpoints-full/<attack>/last.pt.
param(
    [switch]$AllowDisplayOff
)

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class EbbaPower {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS       = [uint32]"0x80000000"
$ES_SYSTEM_REQUIRED  = [uint32]"0x00000001"
$ES_DISPLAY_REQUIRED = [uint32]"0x00000002"

$flags = $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
if (-not $AllowDisplayOff) {
    $flags = $flags -bor $ES_DISPLAY_REQUIRED
}

Write-Host "Requesting keep-awake (Ctrl+C to stop)..." -ForegroundColor Cyan
if ($AllowDisplayOff) {
    Write-Host "  display may turn off; system will stay awake"
} else {
    Write-Host "  system and display will stay awake"
}

[void][EbbaPower]::SetThreadExecutionState($flags)
try {
    while ($true) {
        # Refresh periodically in case another process resets the state.
        Start-Sleep -Seconds 30
        [void][EbbaPower]::SetThreadExecutionState($flags)
    }
}
finally {
    [void][EbbaPower]::SetThreadExecutionState($ES_CONTINUOUS)
    Write-Host "`nKeep-awake released."
}
