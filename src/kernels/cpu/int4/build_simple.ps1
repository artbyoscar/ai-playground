# Simple build script for Windows
# Run from: src/kernels/cpu/int4

Write-Host "🔨 Building EdgeMind Python module" -ForegroundColor Green

# Step 1: Build the core kernels library
Write-Host "`nStep 1: Building kernels..." -ForegroundColor Cyan

if (!(Test-Path build)) {
    mkdir build
}

cd build

# Use Ninja if available, otherwise use default
cmake -DCMAKE_BUILD_TYPE=Release -DINT4_FUSE_BIAS=ON ..
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake failed!" -ForegroundColor Red
    exit 1
}

cmake --build . --config Release
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

cd ..

# Step 2: Find the compiled library
Write-Host "`nStep 2: Finding compiled library..." -ForegroundColor Cyan

$libPath = $null
$possiblePaths = @(
    "build\Release\qgemm_int4.lib",
    "build\qgemm_int4.lib",
    "build\libqgemm_int4.a",
    "build\Debug\qgemm_int4.lib"
)

foreach ($path in $possiblePaths) {
    if (Test-Path $path) {
        $libPath = $path
        Write-Host "Found library: $libPath" -ForegroundColor Green
        break
    }
}

if (!$libPath) {
    Write-Host "Could not find compiled library!" -ForegroundColor Red
    Write-Host "Contents of build directory:" -ForegroundColor Yellow
    Get-ChildItem -Recurse build *.lib,*.a | Select-Object FullName
    exit 1
}

# Step 3: Build Python module
Write-Host "`nStep 3: Building Python module..." -ForegroundColor Cyan

# Get Python paths
$pythonExe = python -c "import sys; print(sys.executable)"
$pythonInc = python -c "import sysconfig; print(sysconfig.get_path('include'))"
$pythonLibDir = python -c "import sysconfig; import os; print(os.path.dirname(sysconfig.get_config_var('LIBRARY')))"
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"

Write-Host "Python: $pythonExe"
Write-Host "Include: $pythonInc"
Write-Host "Lib dir: $pythonLibDir"

# Try with clang first (usually faster compile)
$clangCmd = @"
clang++ -O3 -march=native -mavx2 -mfma -mf16c -shared `
    -I"$pythonInc" `
    -DPy_LIMITED_API `
    simple_bindings.cpp `
    "$libPath" `
    -o edgemind_core.pyd `
    -L"$pythonLibDir" -lpython$pythonVersion
"@

Write-Host "Trying clang..." -ForegroundColor Yellow
Invoke-Expression $clangCmd 2>$null

if (!(Test-Path edgemind_core.pyd)) {
    Write-Host "Clang failed, trying MSVC..." -ForegroundColor Yellow
    
    # Try with MSVC
    $msvcCmd = @"
cl /O2 /MD /I"$pythonInc" /DPy_LIMITED_API `
   simple_bindings.cpp "$libPath" `
   /link /DLL /OUT:edgemind_core.pyd `
   /LIBPATH:"$pythonLibDir" python$pythonVersion.lib
"@
    
    Invoke-Expression $msvcCmd
}

# Step 4: Test the module
if (Test-Path edgemind_core.pyd) {
    Write-Host "`n✅ Module built successfully!" -ForegroundColor Green
    Write-Host "`nTesting module..." -ForegroundColor Cyan
    
    python -c @"
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import edgemind_core
    print('✅ Module imported successfully!')
    print(f'Available functions: {dir(edgemind_core)}')
except Exception as e:
    print(f'❌ Import failed: {e}')
"@
    
    # Copy to parent directory for easier access
    Copy-Item edgemind_core.pyd ..\..\..\ -Force
    Write-Host "`nModule copied to project root" -ForegroundColor Green
    
} else {
    Write-Host "`n❌ Failed to build module" -ForegroundColor Red
    Write-Host "Check error messages above" -ForegroundColor Yellow
}