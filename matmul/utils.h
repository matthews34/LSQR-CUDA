#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <assert.h>

// some garbage collection
#define FREE(ptr) if(ptr) free(ptr)
#define CUDAFREE(ptr) if(ptr) cudaFree(ptr)
#define CUSPARSEFREEHANDLE(ptr) if(ptr) cusparseDestroy(ptr)
#define CUSPARSEFREEMATRIX(ptr) if(ptr) cusparseDestroyMatDescr(ptr)
#define CUDASUCCESS(stat) assert(stat == cudaSuccess)
#define CUSPARSESUCCESS(stat) assert(stat == CUSPARSE_STATUS_SUCCESS)

// status (only for testing)
cudaError_t cudaStat = cudaSuccess;
cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;

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