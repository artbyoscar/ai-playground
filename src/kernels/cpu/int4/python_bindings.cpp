// src/kernels/cpu/int4/python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "model_interface.h"

namespace py = pybind11;

// Python-accessible wrapper for your kernels
class PyEdgeMindKernel {
private:
    edgemind::ModelKernel kernel;
    
public:
    py::array_t<float> gemm_int8(
        py::array_t<float> A,
        py::array_t<int8_t> B_quant,
        py::array_t<float> scales,
        int M, int N, int K
    ) {
        // Get raw pointers
        auto A_ptr = static_cast<float*>(A.mutable_unchecked<2>().mutable_data(0, 0));
        auto B_ptr = static_cast<int8_t*>(B_quant.mutable_unchecked<2>().mutable_data(0, 0));
        auto scales_ptr = static_cast<float*>(scales.mutable_unchecked<1>().mutable_data(0));
        
        // Allocate output
        py::array_t<float> C({M, N});
        auto C_ptr = static_cast<float*>(C.mutable_unchecked<2>().mutable_data(0, 0));
        
        // Call YOUR kernel!
        kernel.gemm_int8(A_ptr, B_ptr, scales_ptr, C_ptr, M, N, K);
        
        return C;
    }
};

PYBIND11_MODULE(edgemind_core, m) {
    m.doc() = "EdgeMind high-performance kernels";
    
    py::class_<PyEdgeMindKernel>(m, "EdgeMindKernel")
        .def(py::init<>())
        .def("gemm_int8", &PyEdgeMindKernel::gemm_int8,
             "High-performance INT8 GEMM (125+ GFLOP/s)");
}