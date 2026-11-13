param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8089,
    [string]$Device = 'Vulkan0',
    [ValidateRange(1, 8)]
    [int]$Slots = 1,
    [ValidateRange(1024, 8192)]
    [int]$ContextPerSlot = 4096
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $root 'models/llama-local/llama-server.exe'
$model = Join-Path $root 'models/llm/Qwen3-4B-Q4_K_M.gguf'

foreach ($path in @($runtime, $model)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing local model asset: $path"
    }
}

& $runtime -m $model --host 127.0.0.1 --port $Port --alias qwen3-4b --device $Device -ngl 99 -c ($ContextPerSlot * $Slots) -np $Slots --jinja --reasoning off
exit $LASTEXITCODE
