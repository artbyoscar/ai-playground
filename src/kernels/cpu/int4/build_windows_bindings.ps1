# Build EdgeMind Python bindings on Windows
# Run from: src/kernels/cpu/int4

Write-Host "🔨 Building EdgeMind Python bindings for Windows" -ForegroundColor Green

# Get Python paths
$pythonPath = python -c "import sys; print(sys.executable)"
$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')"
$pythonInclude = python -c "import sysconfig; print(sysconfig.get_path('include'))"
$pybindIncludes = python -m pybind11 --includes

Write-Host "Python: $pythonPath"
Write-Host "Version: Python $pythonVersion"

# Step 1: Create minimal bindings (no model_interface.cpp yet)
$bindingsCode = @'
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <immintrin.h>
#include <cstring>

namespace py = pybind11;

// Simplified Q8 GEMM kernel for testing
void simple_q8_gemm(
    const float* A,
    const int8_t* B_q8,
    const float* scales,
    float* C,
    int M, int N, int K
) {
    // Very simple implementation just to prove it works
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // Dequantize B on the fly
                float b_val = B_q8[k * N + n] * scales[n];
                sum += A[m * K + k] * b_val;
            }
            C[m * N + n] = sum;
        }
    }
}

// Python wrapper
py::array_t<float> py_q8_gemm(
    py::array_t<float> A,
    py::array_t<int8_t> B_q8,
    py::array_t<float> scales,
    int M, int N, int K
) {
    auto A_buf = A.request();
    auto B_buf = B_q8.request();
    auto S_buf = scales.request();
    
    py::array_t<float> C({M, N});
    auto C_buf = C.request();
    
    simple_q8_gemm(
        static_cast<float*>(A_buf.ptr),
        static_cast<int8_t*>(B_buf.ptr),
        static_cast<float*>(S_buf.ptr),
        static_cast<float*>(C_buf.ptr),
        M, N, K
    );
    
    return C;
}

PYBIND11_MODULE(edgemind_core, m) {
    m.doc() = "EdgeMind kernels (simplified for Windows)";
    m.def("q8_gemm", &py_q8_gemm, "Simple Q8 GEMM");
}
'@

$bindingsCode | Out-File -FilePath "simple_bindings.cpp" -Encoding UTF8

Write-Host "`n📝 Created simple_bindings.cpp"

# Step 2: Compile for Windows (no -fPIC, use MSVC-compatible flags)
Write-Host "`n🔧 Compiling with clang for Windows..."

$compileCmd = @"
clang++ -O3 -march=native -mavx2 -mfma -mf16c ``
    -shared ``
    -I"$pythonInclude" ``
    $pybindIncludes ``
    simple_bindings.cpp ``
    -o edgemind_core.pyd ``
    -l"python$pythonVersion"
"@

Write-Host "Command: $compileCmd"
Invoke-Expression $compileCmd

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Clang failed, trying with cl.exe (MSVC)..." -ForegroundColor Yellow
    
    # Try with MSVC instead
    $vsPath = "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $vsPath) {
        cmd /c "`"$vsPath`" && cl /O2 /MD /I`"$pythonInclude`" $pybindIncludes simple_bindings.cpp /link /DLL /OUT:edgemind_core.pyd python$pythonVersion.lib"
    } else {
        Write-Host "❌ Visual Studio not found. Install VS2022 Build Tools." -ForegroundColor Red
    }
}

# Step 3: Test the module
if (Test-Path "edgemind_core.pyd") {
    Write-Host "`n✅ Module built successfully!" -ForegroundColor Green
    
    Write-Host "`n🧪 Testing module..."
    python -c @"
import numpy as np
import edgemind_core

# Test data
A = np.random.randn(4, 8).astype(np.float32)
B_q8 = np.random.randint(-127, 127, (8, 4), dtype=np.int8)
scales = np.ones(4, dtype=np.float32) * 0.01

# Run kernel
C = edgemind_core.q8_gemm(A, B_q8, scales, 4, 4, 8)
print(f'Success! Output shape: {C.shape}')
print(f'Output values: {C[0, :]}')
"@
} else {
    Write-Host "`n❌ Build failed. Check error messages above." -ForegroundColor Red
}