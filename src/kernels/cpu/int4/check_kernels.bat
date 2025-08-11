@echo off
REM Check if kernels actually exist in compiled libraries
REM Run from: src/kernels/cpu/int4

echo ========================================
echo Checking for EdgeMind kernel symbols...
echo ========================================
echo.

REM Check for compiled libraries
echo Looking for compiled libraries...

if exist build\Release\qgemm_int4.lib (
    echo [OK] Found: build\Release\qgemm_int4.lib
    dumpbin /symbols build\Release\qgemm_int4.lib 2>nul | findstr qgemm_q8_mt >nul
    if %errorlevel% equ 0 (
        echo [OK] Found qgemm_q8_mt symbol!
    ) else (
        echo [!!] qgemm_q8_mt symbol NOT found
    )
) else if exist build\qgemm_int4.lib (
    echo [OK] Found: build\qgemm_int4.lib
    dumpbin /symbols build\qgemm_int4.lib 2>nul | findstr qgemm_q8_mt >nul
    if %errorlevel% equ 0 (
        echo [OK] Found qgemm_q8_mt symbol!
    ) else (
        echo [!!] qgemm_q8_mt symbol NOT found
    )
) else (
    echo [!!] No compiled library found!
    echo      Run: cmake --build build --config Release
)

echo.
echo Looking for object files...

if exist build\*.obj (
    echo [OK] Found object files
    dir /b build\*.obj
)

if exist build\Release\*.obj (
    echo [OK] Found Release object files
    dir /b build\Release\*.obj
)

echo.
echo ========================================
echo Quick kernel test...
echo ========================================

REM Try to link a simple test
echo #include ^<stdio.h^> > test_link.c
echo extern void qgemm_q8_mt(const float*, const signed char*, const float*, const float*, float*, int, int, int, int, int); >> test_link.c
echo int main() { printf("Kernel exists\n"); return 0; } >> test_link.c

cl test_link.c build\Release\qgemm_int4.lib /Fe:test_link.exe 2>nul
if exist test_link.exe (
    echo [OK] Successfully linked to kernel!
    test_link.exe
    del test_link.exe
) else (
    cl test_link.c build\qgemm_int4.lib /Fe:test_link.exe 2>nul
    if exist test_link.exe (
        echo [OK] Successfully linked to kernel!
        test_link.exe
        del test_link.exe
    ) else (
        echo [!!] Cannot link to kernel library
    )
)

del test_link.c 2>nul
del test_link.obj 2>nul

echo.
echo ========================================
echo Summary:
echo ========================================

if exist edgemind_core.pyd (
    echo [OK] Python module exists: edgemind_core.pyd
) else (
    echo [!!] Python module not built yet
    echo      Run: build_ultra_simple.bat
)

echo.
pause