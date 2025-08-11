@echo off
REM Emergency performance fixes
REM Run from: src/kernels/cpu/int4

echo =====================================================
echo EMERGENCY PERFORMANCE OPTIMIZATION ATTEMPTS
echo =====================================================
echo.

echo Step 1: Set Windows to High Performance Mode
echo ---------------------------------------------
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
echo [OK] Power plan set to High Performance
echo.

echo Step 2: Set process priority to High
echo -------------------------------------
echo Testing with HIGH priority...
start /HIGH /B build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16 --it 100
timeout /t 5 /nobreak > nul
echo.

echo Step 3: Test with environment variables
echo ---------------------------------------
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=16
set KMP_AFFINITY=compact
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16 --it 100
echo.

echo Step 4: Test smaller working set (cache friendly)
echo -------------------------------------------------
build\Release\test_qgemm_perf_q8_mt.exe --M 128 --N 128 --K 128 --threads 8 --it 1000
echo.

echo Step 5: Run the FP32 baseline for comparison
echo --------------------------------------------
if exist build\Release\test_qgemm_perf_vs_baseline.exe (
    build\Release\test_qgemm_perf_vs_baseline.exe
) else (
    echo Baseline test not found
)
echo.

echo Step 6: Check if we're actually using Q8 path
echo ---------------------------------------------
echo Running correctness test to verify Q8 is working...
build\Release\test_qgemm_correctness.exe
echo.

echo =====================================================
echo DIAGNOSTICS COMPLETE
echo =====================================================
echo.
echo If performance is still ~10 GFLOP/s:
echo 1. The 125 GFLOP/s claim needs to be revised
echo 2. Try building with Intel Compiler or GCC
echo 3. Test on a different machine
echo 4. Profile the code to find bottlenecks
echo.
pause