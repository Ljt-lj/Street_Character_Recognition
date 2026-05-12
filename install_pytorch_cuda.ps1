# Install PyTorch with CUDA 12.6 (NVIDIA GPU + driver required)
#
# IMPORTANT (WinError 32 "file in use"):
#   Close Cursor, VS Code, Jupyter, and any `python` process before running.
#   Prefer: right-click PowerShell -> Run as administrator, then run this script.
#
# Wheel size ~2.6 GB; unstable network may need retries or mobile hotspot.
#
# Usage:  powershell -ExecutionPolicy Bypass -File .\install_pytorch_cuda.ps1

$ErrorActionPreference = "Stop"

# Use a fresh temp dir to reduce antivirus / indexer locking pip-unpack files
$pipTmp = Join-Path $env:TEMP "pip_torch_install_$(Get-Random)"
New-Item -ItemType Directory -Path $pipTmp -Force | Out-Null
$env:TMP = $pipTmp
$env:TEMP = $pipTmp

Write-Host "Using temp: $pipTmp"

# pip may write "Skipping ... not installed" to stderr; with $ErrorActionPreference Stop
# PowerShell would treat that as a terminating error — relax for uninstall only.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& python -m pip uninstall -y torch torchvision 2>&1 | Out-Null
& python -m pip uninstall -y torchaudio 2>&1 | Out-Null
$ErrorActionPreference = $prevEap

# --no-cache-dir avoids huge partial cache; install to site-packages directly
& python -m pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 `
  --index-url https://download.pytorch.org/whl/cu126

& python -c "import torch; print('version:', torch.__version__); print('cuda built:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count())"

Remove-Item -Recurse -Force $pipTmp -ErrorAction SilentlyContinue
Write-Host "Done."
