#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "matrix.h"

#define PRINT_VECTOR(v, n) std::cout << "[";            \
    for (i = 0; i < n - 1; i++)                         \
    {                                                   \
        std::cout << v[i] << ", ";                      \
    }                                                   \
    std::cout << v[i] << "]" << std::endl;              \

int main(int argc, char **argv)
{
	std::cout << "LSQR-CUDA" << std::endl;

    cusparseStatus_t status;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;
    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
    int i;

    // initialize CUSPARSE library
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "CUSPARSE library initialization failed" << std::endl;
        return 1;
    }
    // create and setup matrix descriptor
    status = cusparseCreateMatDescr(&descr);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::cout << "Matrix descriptor initialization failed" << std::endl;
        return 1;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    // m x n matrix A, n vector x
    int m = 4, n = 5;
    int nnz = 9; // number of nonzero elements of A
    int blockDim = 2; // BSR block dimension (blockDim > 1 for bsrmv)
    // matrix
    float matrixVal[] = {1.0, 4.0, 2.0, 3.0, 5.0, 7.0, 8.0, 9.0, 6.0};
    int row[] = {0, 2, 4, 7, 9};
    int colInd[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    // vector
    float vector[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // allocate and transer vector to GPU 
    float *inGPUvector;
    cudaMalloc((void**)&inGPUvector, sizeof(float) * n);
    cudaMemcpy(inGPUvector, vector, sizeof(float) * n, cudaMemcpyHostToDevice); 
    // allocate output vectors
    float *outCPUvector, *outGPUvector;
    outCPUvector = (float*)malloc(sizeof(float) * m);
    cudaMalloc((void**)&outGPUvector, sizeof(float) * m);
    
    // allocate CPU matrix (in CSR)
    csrCPUMatrix cpuMatrix = matrix_alloc_cpu(m, nnz);
    memcpy(cpuMatrix.csrVal, matrixVal, nnz * sizeof(float));
    memcpy(cpuMatrix.csrRowPtr, row, (m + 1) * sizeof(int));
    memcpy(cpuMatrix.csrColInd, colInd, nnz * sizeof(int));
    
    // transfer matrix to GPU
    csrGPUMatrix csrMatrix = matrix_alloc_gpu(m, nnz);
    matrix_upload(cpuMatrix, csrMatrix, m, nnz);  
    
    // convert CSR matrix to BSR
    bsrGPUMatrix bsrMatrix = bsrFromCsr(handle, dir, descr,
        csrMatrix,
        m, n, blockDim);

    // multiply (y = alpha * A * x + beta * y)
    float alpha = 1, beta = 0; // y = A * x
    mvOperation(handle, dir, descr,
        &alpha, bsrMatrix, inGPUvector, &beta, outGPUvector);
    // cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     bsrMatrix.mb, bsrMatrix.nb, bsrMatrix.nnzb, 
    //     &alpha, descr, bsrMatrix.bsrVal, bsrMatrix.bsrRowPtr, bsrMatrix.bsrColInd, bsrMatrix.blockDim,
    //     inGPUvector, &beta, outGPUvector);

    // transfer output vector to CPU
    cudaMemcpy(outCPUvector, outGPUvector, sizeof(float) * m, cudaMemcpyDeviceToHost); 

    PRINT_VECTOR(outCPUvector, 4);

    // clean up
    matrix_free_cpu(cpuMatrix);
    matrix_free_gpu_csr(csrMatrix);
    matrix_free_gpu_bsr(bsrMatrix);
    cudaFree(inGPUvector);
    cudaFree(outGPUvector);
}
