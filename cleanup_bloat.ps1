# EdgeMind Cleanup Script - Reduces 14GB to <500MB
# Run from: C:\Users\OscarNuñez\Desktop\ai-playground

Write-Host "🧹 EdgeMind Cleanup - Removing 13GB of bloat" -ForegroundColor Green
Write-Host "=" * 60

# Get initial size
$initialSize = (Get-ChildItem -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Initial size: $([math]::Round($initialSize, 2)) GB" -ForegroundColor Yellow

# Step 1: Move models to separate location (keep them, but outside project)
Write-Host "`n📦 Moving models to C:\EdgeMindModels..." -ForegroundColor Cyan
$modelDest = "C:\EdgeMindModels"
if (!(Test-Path $modelDest)) {
    New-Item -ItemType Directory -Path $modelDest | Out-Null
}

# Move all GGUF files
Get-ChildItem -Path "models" -Filter "*.gguf" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Moving $($_.Name) ($('{0:N2}' -f ($_.Length/1GB)) GB)"
    Move-Item $_.FullName -Destination $modelDest -Force
}

# Step 2: Remove the bloated virtual environment (3.5GB of PyTorch we don't need)
Write-Host "`n🐍 Removing bloated virtual environment..." -ForegroundColor Cyan
if (Test-Path "ai-env") {
    Remove-Item -Recurse -Force "ai-env" -ErrorAction SilentlyContinue
    Write-Host "  Removed ai-env (saved ~3.5 GB)"
}

# Step 3: Remove DirectML stuff (500MB we're not using)
Write-Host "`n🗑️ Removing DirectML third-party files..." -ForegroundColor Cyan
if (Test-Path "third_party") {
    # Keep README if exists
    $readmes = Get-ChildItem -Path "third_party" -Filter "README.md" -Recurse
    $tempReadmes = @()
    foreach ($readme in $readmes) {
        $tempPath = Join-Path $env:TEMP $readme.Name
        Copy-Item $readme.FullName $tempPath
        $tempReadmes += @{Original = $readme.FullName; Temp = $tempPath}
    }
    
    Remove-Item -Recurse -Force "third_party/*" -ErrorAction SilentlyContinue
    
    # Restore READMEs
    foreach ($readme in $tempReadmes) {
        $dir = Split-Path $readme.Original -Parent
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
        Move-Item $readme.Temp $readme.Original -Force
    }
    
    Write-Host "  Removed DirectML files (saved ~500 MB)"
}

# Step 4: Remove build artifacts
Write-Host "`n🔨 Removing build artifacts..." -ForegroundColor Cyan
$buildDirs = @(
    "build",
    "cmake-build-*",
    "src/kernels/cpu/int4/build*",
    "src/kernels/cpu/int4/pack"
)

foreach ($pattern in $buildDirs) {
    Get-ChildItem -Path . -Filter $pattern -Recurse -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        Remove-Item -Recurse -Force $_.FullName
        Write-Host "  Removed $($_.Name)"
    }
}

# Step 5: Clean Python cache
Write-Host "`n🧹 Cleaning Python cache..." -ForegroundColor Cyan
Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
Get-ChildItem -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force

# Step 6: Remove the 5MB nuget.exe
if (Test-Path "nuget.exe") {
    Remove-Item "nuget.exe" -Force
    Write-Host "  Removed nuget.exe"
}

# Step 7: Create clean virtual environment (minimal, no PyTorch)
Write-Host "`n✨ Creating minimal virtual environment..." -ForegroundColor Cyan
python -m venv edgemind-env
Write-Host "  Created edgemind-env (minimal)"

# Step 8: Update .gitignore
Write-Host "`n📝 Updating .gitignore..." -ForegroundColor Cyan
$gitignore = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.egg-info/
dist/
venv/
ai-env/
edgemind-env/
.env
*.pyd
*.so

# Models (keep outside repo)
models/
*.gguf
*.ggml
*.bin
*.safetensors
*.pt
*.pth

# Build artifacts
build/
cmake-build-*/
*.o
*.obj
*.exe
*.dll
*.lib
*.a
*.exp
*.ilk
*.idb
*.pdb
*.ninja*
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
CTestTestfile.cmake
Testing/

# Third-party
third_party/
nuget.exe

# IDE
.vscode/
.idea/
*.swp
*.swo
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Logs and temp
*.log
*.tmp
temp/
tmp/

# Data
data/large/
*.h5
*.hdf5
*.pkl
*.pickle
*.db
*.sqlite
"@

$gitignore | Out-File -FilePath ".gitignore" -Encoding UTF8
Write-Host "  Updated .gitignore"

# Get final size
$finalSize = (Get-ChildItem -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
$saved = $initialSize - $finalSize

Write-Host "`n" + ("=" * 60) -ForegroundColor Green
Write-Host "✅ CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "  Initial size: $('{0:N2}' -f $initialSize) GB" -ForegroundColor Yellow
Write-Host "  Final size:   $('{0:N2}' -f $finalSize) GB" -ForegroundColor Green
Write-Host "  Space saved:  $('{0:N2}' -f $saved) GB" -ForegroundColor Cyan
Write-Host "`n📦 Models moved to: $modelDest" -ForegroundColor Cyan
Write-Host "💡 Next: Activate edgemind-env and install minimal dependencies" -ForegroundColor Magenta