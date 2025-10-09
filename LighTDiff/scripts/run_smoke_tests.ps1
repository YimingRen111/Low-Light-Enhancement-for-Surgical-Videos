param(
    [string]$envName = "basicsr-env"
)

Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir "..")
$basicSrPath = Join-Path $repoRoot "BasicSR"

Write-Host "Activating conda env: $envName"
conda activate $envName

Write-Host "Ensuring BasicSR import works (python -c)..."
python -c "import sys; sys.path.insert(0, r'$basicSrPath'); import basicsr; print('OK', basicsr.__version__)"

Write-Host "Checking tests data paths and generating placeholder images if missing..."
$testsData = Join-Path $basicSrPath "tests\data"
if (-Not (Test-Path $testsData)) {
    Write-Host "tests/data not found. Creating and adding placeholders..."
    New-Item -ItemType Directory -Force -Path $testsData | Out-Null
}

$bicDir = Join-Path $testsData "bic"
$gtDir = Join-Path $testsData "gt"
if (-Not (Test-Path $bicDir)) { New-Item -ItemType Directory -Force -Path $bicDir | Out-Null }
if (-Not (Test-Path $gtDir)) { New-Item -ItemType Directory -Force -Path $gtDir | Out-Null }

Write-Host "Generating small placeholder images (if they don't exist)"
python (Join-Path $scriptDir 'generate_placeholder_images.py') --out1 (Join-Path $bicDir 'baboon.png') --out2 (Join-Path $gtDir 'baboon.png')
python (Join-Path $scriptDir 'generate_placeholder_images.py') --out1 (Join-Path $bicDir 'comic.png') --out2 (Join-Path $gtDir 'comic.png')

Write-Host "Running metrics smoke test script"
python (Join-Path $basicSrPath 'basicsr\metrics\test_metrics\test_psnr_ssim.py')

Write-Host "Smoke tests finished. If you want to run more, consider running specific scripts or pytest in the BasicSR package." 
