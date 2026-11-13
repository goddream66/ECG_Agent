param(
    [ValidatePattern('^\d+_\d{2}-\d{2}-\d{2}$')]
    [string[]]$Record = @('1_10-09-54', '1_10-11-48')
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $root 'data/senssmarttech'
$source = 'https://physionet.org/files/senssmarttech/1.0.0'
New-Item -ItemType Directory -Path (Join-Path $destination 'WFDB') -Force | Out-Null

$files = @('Demographics.csv', 'LICENSE.txt')
foreach ($id in $Record) {
    foreach ($modality in @('ecg', 'ppg', 'pcg')) {
        foreach ($extension in @('hea', 'dat')) {
            $files += "WFDB/${id}_${modality}.$extension"
        }
    }
}

foreach ($file in $files) {
    $path = Join-Path $destination $file
    if ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -gt 0) { continue }
    Invoke-WebRequest -Uri "$source/$file" -OutFile "$path.download" -TimeoutSec 60
    Move-Item -LiteralPath "$path.download" -Destination $path -Force
    Write-Output $file
}
