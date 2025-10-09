# run_with_env.ps1
# Helper script to run any python command inside the `lightdiff` conda env with correct PYTHONPATH and PATH fixes.
param(
    [Parameter(Mandatory=$true)] [string]$CmdLine
)

# Ensure System32 in PATH
if (-not ($env:PATH -match "\\\bSystem32\\\b")) {
    $env:PATH = "$env:windir\System32;$env:PATH"
    Write-Host "Prepended System32 to PATH"
}

# Set PYTHONPATH for repo packages
# Prefer the conda env's Library\bin so its OpenMP runtime is used first
$condaEnvPath = "$env:USERPROFILE\\anaconda3\\envs\\lightdiff"
$envLibBin = Join-Path $condaEnvPath "Library\bin"
if (Test-Path $envLibBin) {
    $env:PATH = "$envLibBin;$env:PATH"
    Write-Host "Prepended $envLibBin to PATH"
} else {
    # Fallback: try CONDA_PREFIX if set
    if ($env:CONDA_PREFIX) {
        $envLibBin = Join-Path $env:CONDA_PREFIX "Library\bin"
        if (Test-Path $envLibBin) {
            $env:PATH = "$envLibBin;$env:PATH"
            Write-Host "Prepended $envLibBin to PATH (from CONDA_PREFIX)"
        }
    }
}

# Set PYTHONPATH for repo packages
$env:PYTHONPATH = "E:\ELEC5020\LighTDiff-main\LighTDiff-main\BasicSR;E:\ELEC5020\LighTDiff-main\LighTDiff-main\LighTDiff"

# Reduce thread over-subscription for OpenMP/MKL
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'

# Run the provided command inside conda env
Write-Host "Running in conda env 'lightdiff': $CmdLine"
conda run -n lightdiff --no-capture-output powershell -Command $CmdLine
