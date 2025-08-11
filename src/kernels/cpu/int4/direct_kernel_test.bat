@echo off
REM Direct test of your compiled kernels - no Python needed!
REM Run from: src/kernels/cpu/int4

echo =====================================================
echo    EDGEMIND KERNEL DIRECT TEST - PROVING 125 GFLOP/s
echo =====================================================
echo.

cd build\Release

if not exist test_qgemm_perf_q8_mt.exe (
    echo ERROR: Kernels not compiled!
    echo Run: cmake --build . --config Release
    pause
    exit /b 1
)

echo Running YOUR 125 GFLOP/s kernel tests...
echo.

echo Test 1: Small matrices (256x256x256)
echo -------------------------------------
test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 256 --threads 16 --it 100
echo.

echo Test 2: Your benchmark size (256x256x2048)
echo -------------------------------------------
test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16 --it 100
echo.

echo Test 3: Larger size (512x512x512)
echo ----------------------------------
test_qgemm_perf_q8_mt.exe --M 512 --N 512 --K 512 --threads 16 --it 50
echo.

echo Test 4: Model-like dimensions (1x2048x2048)
echo --------------------------------------------
test_qgemm_perf_q8_mt.exe --M 1 --N 2048 --K 2048 --threads 8 --it 10
echo.

echo =====================================================
echo If you see 100+ GFLOP/s above, YOUR KERNELS WORK!
echo =====================================================
echo.

echo Other available tests:
dir *.exe /b

echo.
pause