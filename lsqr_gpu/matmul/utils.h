#pragma once

#include <cuda.h>
#include <assert.h>
#include <stdlib.h>
#include <string>
#include <string.h> 
#include <stdio.h>

#define BLOCK_SIZE 16
#define ZERO 0.000000001

// some garbage collection
#define FREE(ptr) if(ptr) free(ptr)
#define CUDAFREE(ptr) if(ptr) cudaFree(ptr)
#define CUDASUCCESS(stat) assert(stat == cudaSuccess)
#define CUSPARSESUCCESS(stat) assert(stat == CUSPARSE_STATUS_SUCCESS)

// status (only for testing)
extern cudaError_t cudaStat; //= cudaSuccess;

// loads template array to gpu
template <typename datatype>
void to_gpu(datatype *array_cpu, datatype *array_gpu, const int n) {
	cudaMemcpy(array_gpu,array_cpu,sizeof(datatype)*n,cudaMemcpyHostToDevice);
}

// loads template array from gpu
template <typename datatype>
void from_gpu(datatype *array_cpu, datatype *array_gpu, const int n) {
	cudaMemcpy(array_cpu,array_gpu,sizeof(datatype)*n,cudaMemcpyDeviceToHost);
}