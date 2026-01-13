# Upload data files to RunPod
# Usage: .\upload_to_runpod.ps1
#   Then paste your SSH command when prompted
#
# Or directly: .\upload_to_runpod.ps1 "ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519"

param(
    [string]$SshCommand = ""
)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Hybridyzer RunPod Uploader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get SSH command if not provided
if (-not $SshCommand) {
    Write-Host "Paste your RunPod SSH command (from the Connect button):" -ForegroundColor Yellow
    Write-Host "Example: ssh root@213.192.2.93 -p 40110 -i ~/.ssh/id_ed25519" -ForegroundColor DarkGray
    Write-Host ""
    $SshCommand = Read-Host "SSH command"
}

# Parse the SSH command
# Expected format: ssh root@<IP> -p <PORT> -i <keypath>
if ($SshCommand -match 'root@([\d\.]+).*-p\s*(\d+)') {
    $IP = $Matches[1]
    $PORT = $Matches[2]
} elseif ($SshCommand -match '([\d\.]+):(\d+)') {
    # Also accept IP:PORT format
    $IP = $Matches[1]
    $PORT = $Matches[2]
} else {
    Write-Host "Could not parse SSH command. Expected format:" -ForegroundColor Red
    Write-Host "  ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Detected:" -ForegroundColor Green
Write-Host "  IP:   $IP"
Write-Host "  Port: $PORT"
Write-Host ""

# Find data files
$DataDir = Join-Path $PSScriptRoot "data"
$FilesToUpload = @(
    "btcusd_5min_train_2017_2022.csv",
    "btcusd_5min_val_2023.csv",
    "btcusd_5min_test_2024.csv",
    "btcusd_5min_test_2025.csv"
)

# Check which files exist
$ExistingFiles = @()
foreach ($file in $FilesToUpload) {
    $path = Join-Path $DataDir $file
    if (Test-Path $path) {
        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Write-Host "  Found: $file ($size MB)" -ForegroundColor Green
        $ExistingFiles += $path
    } else {
        Write-Host "  Missing: $file" -ForegroundColor Yellow
    }
}

if ($ExistingFiles.Count -eq 0) {
    Write-Host ""
    Write-Host "No data files found in $DataDir" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Uploading $($ExistingFiles.Count) files to RunPod..." -ForegroundColor Cyan
Write-Host ""

# SSH key path
$SshKey = "$env:USERPROFILE\.ssh\id_ed25519"
if (-not (Test-Path $SshKey)) {
    $SshKey = "$env:USERPROFILE\.ssh\id_rsa"
}

# Upload each file
$success = 0
foreach ($file in $ExistingFiles) {
    $filename = Split-Path $file -Leaf
    Write-Host "Uploading $filename..." -NoNewline
    
    $dest = "root@${IP}:/workspace/Hybridyzer/data/"
    
    # Run scp
    $scpResult = & scp -P $PORT -i $SshKey -o StrictHostKeyChecking=no $file $dest 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host " OK" -ForegroundColor Green
        $success++
    } else {
        Write-Host " FAILED" -ForegroundColor Red
        Write-Host "  $scpResult" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Upload Complete: $success/$($ExistingFiles.Count) files" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next: SSH into RunPod and start training:" -ForegroundColor Yellow
Write-Host "  $SshCommand"
Write-Host "  cd /workspace/Hybridyzer"
Write-Host "  python train.py --runpod --walkforward --gpu"
Write-Host ""
