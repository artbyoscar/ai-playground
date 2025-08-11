# setup.py for pip installable package
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "edgemind_kernels",
        ["python/edgemind_bindings.cpp"],
        include_dirs=["src/kernels/cpu/int4"],
        libraries=["qgemm_int4"],
        cxx_std=17,
    ),
]

setup(
    name="edgemind-kernels",
    version="1.0.0",
    ext_modules=ext_modules,
    description="High-performance INT4/Q8 GEMM kernels",
)