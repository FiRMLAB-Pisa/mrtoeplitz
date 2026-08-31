/**
 * @file bindings.hpp
 * @brief The kernel groups that make up `mrtoeplitz._ext`.
 *
 * Each is defined in its own translation unit; module.cpp creates the module
 * and calls these in turn.
 */

#pragma once

#include <pybind11/pybind11.h>

void mrtoeplitz_bind_packed_matvec(pybind11::module_& module);
