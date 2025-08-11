@echo off
REM Ultra simple build for Windows
REM Run from: src/kernels/cpu/int4

echo Building EdgeMind kernels...

REM Step 1: Build kernels with CMake
if not exist build mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..

REM Step 2: Find Python paths
for /f "delims=" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
for /f "delims=" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INC=%%i
for /f "delims=" %%i in ('python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"') do set PYTHON_LIB=%%i

echo Python: %PYTHON_EXE%
echo Include: %PYTHON_INC%
echo Library: %PYTHON_LIB%

REM Step 3: Try to compile with cl.exe (MSVC)
echo.
echo Compiling Python module with MSVC...

cl /O2 /MD /I"%PYTHON_INC%" ultra_simple_bindings.cpp /link /DLL /OUT:edgemind_core.pyd %PYTHON_LIB%.lib

if exist edgemind_core.pyd (
    echo SUCCESS! Module built.
    echo Testing...
    python -c "import edgemind_core; print('Module loaded!'); print(edgemind_core.test_kernel(32, 32, 32))"
) else (
    echo Build failed. Trying simpler version...
    
    REM Try without linking to kernel library
    cl /O2 /MD /I"%PYTHON_INC%" ultra_simple_bindings.cpp /link /DLL /OUT:edgemind_core.pyd %PYTHON_LIB%.lib /FORCE
)