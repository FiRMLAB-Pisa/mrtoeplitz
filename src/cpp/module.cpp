/**
 * @file module.cpp
 * @brief The compiled kernels, bound as `mrtoeplitz._ext`.
 */

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_ext, module)
{
    module.doc() = "Precompiled kernels for mrtoeplitz";
}
