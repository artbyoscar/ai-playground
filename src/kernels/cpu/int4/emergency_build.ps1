# Emergency build script - tries multiple approaches
# Run from: src/kernels/cpu/int4

Write-Host "`n🚀 Emergency EdgeMind Build Script" -ForegroundColor Green
Write-Host "="*50 -ForegroundColor Green

# Get Python info
$pythonInfo = python -c "import sys, sysconfig; print(f'{sys.executable}|{sysconfig.get_path(\"include\")}|{sys.version_info.major}{sys.version_info.minor}')"
$parts = $pythonInfo -split '\|'
$pythonExe = $parts[0]
$pythonInc = $parts[1]
$pythonVer = $parts[2]

Write-Host "Python: $pythonExe"
Write-Host "Include: $pythonInc"
Write-Host "Version: python$pythonVer"

# Method 1: Try with cl.exe (MSVC)
Write-Host "`n🔧 Method 1: MSVC Compiler..." -ForegroundColor Cyan

$msvcCmd = "cl /O2 /MD /I`"$pythonInc`" ultra_simple_bindings.cpp /link /DLL /OUT:edgemind_core.pyd python$pythonVer.lib 2>&1"
$output = cmd /c $msvcCmd

if (Test-Path "edgemind_core.pyd") {
    Write-Host "✅ MSVC build successful!" -ForegroundColor Green
} else {
    Write-Host "❌ MSVC failed" -ForegroundColor Yellow
    
    # Method 2: Try with clang
    Write-Host "`n🔧 Method 2: Clang Compiler..." -ForegroundColor Cyan
    
    $clangExists = Get-Command clang -ErrorAction SilentlyContinue
    if ($clangExists) {
        clang -shared -O2 -o edgemind_core.pyd `
            -I"$pythonInc" `
            ultra_simple_bindings.cpp `
            -lpython$pythonVer 2>&1
        
        if (Test-Path "edgemind_core.pyd") {
            Write-Host "✅ Clang build successful!" -ForegroundColor Green
        }
    } else {
        Write-Host "Clang not found" -ForegroundColor Yellow
    }
    
    # Method 3: Try with g++ (MinGW)
    if (!(Test-Path "edgemind_core.pyd")) {
        Write-Host "`n🔧 Method 3: G++ Compiler..." -ForegroundColor Cyan
        
        $gppExists = Get-Command g++ -ErrorAction SilentlyContinue
        if ($gppExists) {
            g++ -shared -O2 -o edgemind_core.pyd `
                -I"$pythonInc" `
                ultra_simple_bindings.cpp `
                -lpython$pythonVer 2>&1
            
            if (Test-Path "edgemind_core.pyd") {
                Write-Host "✅ G++ build successful!" -ForegroundColor Green
            }
        } else {
            Write-Host "G++ not found" -ForegroundColor Yellow
        }
    }
}

# Method 4: Last resort - create dummy module
if (!(Test-Path "edgemind_core.pyd")) {
    Write-Host "`n⚠️ All compilers failed. Creating dummy module..." -ForegroundColor Yellow
    
    # Create a minimal Python module in pure Python
    @'
def test_kernel(M, N, K):
    """Dummy kernel for testing"""
    # Just return a simple calculation
    return float(M * N * K * 0.001)

print("WARNING: Using dummy Python module - no real kernels!")
'@ | Out-File -FilePath "edgemind_core.py" -Encoding UTF8
    
    Write-Host "Created edgemind_core.py as fallback" -ForegroundColor Yellow
}

# Test the module
Write-Host "`n🧪 Testing module..." -ForegroundColor Cyan

python -c @"
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import edgemind_core
    result = edgemind_core.test_kernel(32, 32, 32)
    print(f'✅ Module works! Test result: {result}')
except Exception as e:
    print(f'❌ Module test failed: {e}')
"@

Write-Host "`n" + "="*50 -ForegroundColor Green

if (Test-Path "edgemind_core.pyd") {
    Write-Host "✅ Build complete! C++ module created." -ForegroundColor Green
    Write-Host "   Location: $(Get-Location)\edgemind_core.pyd" -ForegroundColor Green
} elseif (Test-Path "edgemind_core.py") {
    Write-Host "⚠️ Using Python fallback module" -ForegroundColor Yellow
    Write-Host "   You need to install a C++ compiler to use real kernels" -ForegroundColor Yellow
} else {
    Write-Host "❌ Build failed completely" -ForegroundColor Red
}