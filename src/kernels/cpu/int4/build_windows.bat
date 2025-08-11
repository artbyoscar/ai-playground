@echo off
REM Build EdgeMind kernels with Python bindings on Windows
REM Run from: src/kernels/cpu/int4

echo Building EdgeMind kernels for Windows...

REM Step 1: Build the core kernels as a static library
echo.
echo Step 1: Building core kernels...
if not exist build mkdir build
cd build

cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release -DINT4_FUSE_BIAS=ON ..
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    exit /b 1
)

ninja
if %errorlevel% neq 0 (
    echo Ninja build failed!
    exit /b 1
)

cd ..

REM Step 2: Build Python extension
echo.
echo Step 2: Building Python extension...

REM Get Python configuration
for /f "delims=" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('include'))"') do set PYTHON_INCLUDE=%%i
for /f "delims=" %%i in ('python -c "import numpy; print(numpy.get_include())"') do set NUMPY_INCLUDE=%%i
for /f "delims=" %%i in ('python -c "import sys; print(f'python{sys.version_info.major}{sys.version_info.minor}')"') do set PYTHON_LIB=%%i

echo Python include: %PYTHON_INCLUDE%
echo NumPy include: %NUMPY_INCLUDE%
echo Python lib: %PYTHON_LIB%

REM Compile with cl.exe (MSVC)
cl /O2 /MD /I"%PYTHON_INCLUDE%" /I"%NUMPY_INCLUDE%" /DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ^
   working_bindings.cpp build\qgemm_int4.lib ^
   /link /DLL /OUT:edgemind_core.pyd %PYTHON_LIB%.lib

if %errorlevel% neq 0 (
    echo.
    echo MSVC failed, trying with clang...
    
    clang++ -O3 -march=native -mavx2 -mfma -mf16c -shared ^
        -I"%PYTHON_INCLUDE%" -I"%NUMPY_INCLUDE%" ^
        -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION ^
        working_bindings.cpp build\qgemm_int4.lib ^
        -o edgemind_core.pyd -l%PYTHON_LIB%
)

if exist edgemind_core.pyd (
    echo.
    echo SUCCESS! edgemind_core.pyd created
    echo Testing module...
    python -c "import edgemind_core; print('Module loaded successfully!')"
) else (
    echo.
    echo ERROR: Failed to create edgemind_core.pyd
    exit /b 1
)

echo.
echo Build complete!
echo Copy edgemind_core.pyd to your Python path to use it.