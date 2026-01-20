# Upload data files to RunPod and optionally run setup
# Usage: .\upload_to_runpod.ps1
#   Then paste your SSH command when prompted
#
# Or directly: .\upload_to_runpod.ps1 "ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519"
# With auto-setup: .\upload_to_runpod.ps1 -Setup "ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519"

param(
    [string]$SshCommand = "",
    [switch]$Setup,
    [switch]$SkipUpload
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
}
elseif ($SshCommand -match '([\d\.]+):(\d+)') {
    # Also accept IP:PORT format
    $IP = $Matches[1]
    $PORT = $Matches[2]
}
else {
    Write-Host "Could not parse SSH command. Expected format:" -ForegroundColor Red
    Write-Host "  ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Detected:" -ForegroundColor Green
Write-Host "  IP:   $IP"
Write-Host "  Port: $PORT"
Write-Host ""

# SSH key path
$SshKey = "$env:USERPROFILE\.ssh\id_ed25519"
if (-not (Test-Path $SshKey)) {
    $SshKey = "$env:USERPROFILE\.ssh\id_rsa"
}

# SSH agent setup - cache passphrase for session
Write-Host "Checking SSH agent..." -ForegroundColor Yellow
$sshAgentProcess = Get-Process ssh-agent -ErrorAction SilentlyContinue
if (-not $sshAgentProcess) {
    Write-Host "Starting SSH agent..." -ForegroundColor Yellow
    Start-Process ssh-agent -WindowStyle Hidden
    Start-Sleep -Seconds 2
}

# Check if key is already loaded in agent
$keyLoaded = $false
try {
    $keys = & ssh-add -l 2>&1
    if ($LASTEXITCODE -eq 0 -and $keys -match [regex]::Escape($SshKey)) {
        Write-Host "SSH key already loaded in agent" -ForegroundColor Green
        $keyLoaded = $true
    }
}
catch {
    # ssh-add might not be available, continue
}

# Add key to agent if not loaded
if (-not $keyLoaded) {
    Write-Host "Adding SSH key to agent (enter passphrase once)..." -ForegroundColor Yellow
    & ssh-add $SshKey
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SSH key added to agent - passphrase cached for this session" -ForegroundColor Green
    }
    else {
        Write-Host "Warning: Failed to add key to agent. You may need to enter passphrase multiple times." -ForegroundColor Yellow
    }
}

# SSH/SCP common options with keepalive to prevent connection drops
# Use array format so PowerShell expands them correctly
$SshCommonOpts = @(
    "-o", "StrictHostKeyChecking=no",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "ConnectTimeout=10"
)

# STEP 1: Ensure repo exists on pod (before upload)
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Preparing Remote Environment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Checking if Hybridyzer is cloned..." -ForegroundColor Yellow
$checkCmd = "if [ -d /workspace/Hybridyzer ]; then echo 'EXISTS'; else echo 'MISSING'; fi"
$result = & ssh -p $PORT -i $SshKey $SshCommonOpts "root@$IP" $checkCmd 2>&1

if ($result -match "MISSING") {
    Write-Host "Cloning repository..." -ForegroundColor Yellow
    $cloneResult = & ssh -p $PORT -i $SshKey $SshCommonOpts "root@$IP" "cd /workspace && git clone https://github.com/jratnieks/Hybridyzer.git" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone repository:" -ForegroundColor Red
        Write-Host $cloneResult -ForegroundColor Red
        exit 1
    }
    Write-Host "Repository cloned successfully" -ForegroundColor Green
}
else {
    Write-Host "Repository exists, pulling latest..." -ForegroundColor Green
    $pullResult = & ssh -p $PORT -i $SshKey $SshCommonOpts "root@$IP" "cd /workspace/Hybridyzer && git pull origin master" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: git pull failed (continuing anyway)" -ForegroundColor Yellow
    }
}

# Ensure data directory exists
Write-Host "Ensuring data directory exists..." -ForegroundColor Yellow
& ssh -p $PORT -i $SshKey $SshCommonOpts "root@$IP" "mkdir -p /workspace/Hybridyzer/data" | Out-Null
Write-Host "Data directory ready" -ForegroundColor Green

$success = 0
$totalFiles = 0

# STEP 2: Upload data files
if (-not $SkipUpload) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Uploading Data Files" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
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
        }
        else {
            Write-Host "  Missing: $file" -ForegroundColor Yellow
        }
    }

    if ($ExistingFiles.Count -eq 0) {
        Write-Host ""
        Write-Host "No data files found in $DataDir" -ForegroundColor Red
        Write-Host "Use -SkipUpload -Setup to just run setup without uploading" -ForegroundColor Yellow
        exit 1
    }

    Write-Host ""
    Write-Host "Uploading $($ExistingFiles.Count) files to RunPod..." -ForegroundColor Cyan
    Write-Host ""

    # Upload each file
    $totalFiles = $ExistingFiles.Count
    foreach ($file in $ExistingFiles) {
        $filename = Split-Path $file -Leaf
        Write-Host "Uploading $filename..." -NoNewline
        
        $dest = "root@${IP}:/workspace/Hybridyzer/data/"
        
        # Run scp with keepalive to prevent connection drops
        $scpResult = & scp -P $PORT -i $SshKey $SshCommonOpts $file $dest 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host " OK" -ForegroundColor Green
            $success++
        }
        else {
            Write-Host " FAILED" -ForegroundColor Red
            Write-Host "  $scpResult" -ForegroundColor Red
        }
    }
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Upload Complete: $success/$totalFiles files" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
}
else {
    Write-Host ""
    Write-Host "Skipping file upload (-SkipUpload flag)" -ForegroundColor Yellow
}

# STEP 3: Run setup if requested
# Ask about setup if not specified
if (-not $Setup -and -not $SkipUpload) {
    Write-Host ""
    $response = Read-Host "Run setup_runpod.sh on the pod? (Y/n)"
    if ($response -eq "" -or $response -match "^[Yy]") {
        $Setup = $true
    }
}

if ($Setup) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Running Remote Setup" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Running setup_runpod.sh..." -ForegroundColor Yellow
    Write-Host ""
    
    # Run setup script (this will show output live)
    & ssh -p $PORT -i $SshKey $SshCommonOpts "root@$IP" "cd /workspace/Hybridyzer && bash setup_runpod.sh"
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Setup Complete!" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Ready to train! SSH in and run:" -ForegroundColor Green
    Write-Host "  $SshCommand"
    Write-Host "  cd /workspace/Hybridyzer"
    Write-Host "  python train.py --runpod --walkforward"
    Write-Host ""
}
else {
    Write-Host ""
    Write-Host "Next: SSH into RunPod and start training:" -ForegroundColor Yellow
    Write-Host "  $SshCommand"
    Write-Host "  cd /workspace/Hybridyzer"
    if (-not $SkipUpload) {
        Write-Host "  bash setup_runpod.sh  # if first time"
    }
    Write-Host "  python train.py --runpod --walkforward"
    Write-Host ""
}