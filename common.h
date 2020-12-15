#pragma once

#include <iostream>

#include <cuda_runtime.h>

#include "matrix.h"

#define CUDA_CHECK_ERROR                                                       \
    do {                                                                       \
        const cudaError_t err = cudaGetLastError();                            \
        if (err != cudaSuccess) {                                              \
            const char *const err_str = cudaGetErrorString(err);               \
            std::cerr << "Cuda error in " << __FILE__ << ":" << __LINE__ - 1   \
                      << ": " << err_str << " (" << err << ")" << std::endl;   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while(0)


inline unsigned int div_up(unsigned int numerator, unsigned int denominator)
{
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
	return result;
}
