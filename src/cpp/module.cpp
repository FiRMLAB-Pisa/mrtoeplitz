/**
 * @file module.cpp
 * @brief The compiled kernels, bound as `mrtoeplitz._ext`.
 */

#include <pybind11/pybind11.h>

#include "bindings.hpp"

PYBIND11_MODULE(_ext, module)
{
    module.doc() = "Precompiled CPU kernels for mrtoeplitz";
    mrtoeplitz_bind_packed_matvec(module);
}
