@echo off
REM Rebuild kernels with MAXIMUM optimizations
REM Run from: src/kernels/cpu/int4

echo =====================================================
echo REBUILDING WITH MAXIMUM OPTIMIZATIONS
echo =====================================================
echo.

REM Clean old build
echo Cleaning old build...
if exist build rmdir /s /q build
mkdir build
cd build

REM Configure with aggressive optimizations
echo.
echo Configuring with AVX2 optimizations...
echo --------------------------------------

REM Try with Ninja and Clang (usually faster)
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ^
    -DCMAKE_CXX_FLAGS="/O2 /arch:AVX2 /fp:fast /GL /Oi /Ot /Qpar /Qvec-report:2" ^
    -DINT4_FUSE_BIAS=ON ..

if %errorlevel% neq 0 (
    echo Ninja/Clang failed, trying MSVC...
    cmake -G "Visual Studio 17 2022" -A x64 ^
        -DCMAKE_BUILD_TYPE=Release ^
        -DCMAKE_CXX_FLAGS_RELEASE="/O2 /Ob2 /arch:AVX2 /fp:fast /GL /Oi /Ot" ^
        -DINT4_FUSE_BIAS=ON ..
)

echo.
echo Building with optimizations...
echo ------------------------------
cmake --build . --config Release --parallel

cd ..

echo.
echo Testing optimized build...
echo -------------------------
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16 --it 10

echo.
echo =====================================================
echo If performance is still low, try:
echo 1. Disable Windows Defender real-time scanning
echo 2. Set Power Plan to High Performance
echo 3. Close other applications
echo =====================================================
pause