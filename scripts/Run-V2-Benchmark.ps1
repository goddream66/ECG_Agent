param(
    [ValidateSet('train', 'validation', 'test', 'all')]
    [string]$Split = 'validation',
    [switch]$MissingModalities,
    [switch]$IncludeAgents,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $options = @('benchmark', '--split', $Split, '--mode', 'v1_fixed', '--mode', 'v2_events')
    if ($Limit -lt 0) { throw 'Limit must be nonnegative.' }
    if ($Limit -gt 0) { $options += @('--limit', $Limit) }
    if ($MissingModalities) {
        $options += @('--scenario', 'clean', '--scenario', 'drop_ecg', '--scenario', 'drop_ppg', '--scenario', 'drop_pcg')
    }
    if ($IncludeAgents) {
        $options += @('--mode', 'v1_single', '--mode', 'v1_multi', '--with-models')
    }
    & .\.venv\Scripts\python.exe -m cardioagent.cli @options
    if ($LASTEXITCODE -ne 0) { throw "Benchmark exited with code $LASTEXITCODE." }
} finally {
    Pop-Location
}
