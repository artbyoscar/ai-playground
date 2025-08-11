@echo off
REM Check how kernels were actually compiled
REM Run from: src/kernels/cpu/int4

echo =====================================================
echo CHECKING BUILD CONFIGURATION
echo =====================================================
echo.

echo Checking CMakeCache for optimization flags...
echo ---------------------------------------------
findstr /C:"CMAKE_BUILD_TYPE" build\CMakeCache.txt
findstr /C:"CMAKE_CXX_FLAGS" build\CMakeCache.txt
findstr /C:"AVX" build\CMakeCache.txt
findstr /C:"arch:AVX2" build\CMakeCache.txt
echo.

echo Checking if AVX2 is being used...
echo ---------------------------------
dumpbin /disasm build\Release\test_qgemm_perf_q8_mt.exe | findstr /C:"vfmadd" > nul
if %errorlevel% equ 0 (
    echo [OK] FMA instructions found
) else (
    echo [!!] No FMA instructions - NOT using AVX2!
)

dumpbin /disasm build\Release\test_qgemm_perf_q8_mt.exe | findstr /C:"vperm" > nul
if %errorlevel% equ 0 (
    echo [OK] AVX2 permute instructions found
) else (
    echo [!!] No AVX2 instructions found!
)

echo.
echo Testing different thread counts...
echo ----------------------------------
echo.
echo 1 thread:
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 1 --it 10
echo.
echo 4 threads:
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 4 --it 10
echo.
echo 8 threads:
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 8 --it 10
echo.
echo 16 threads:
build\Release\test_qgemm_perf_q8_mt.exe --M 256 --N 256 --K 2048 --threads 16 --it 10
echo.

echo Checking CPU capabilities...
echo ---------------------------
wmic cpu get name, numberofcores, numberoflogicalprocessors
echo.

pause