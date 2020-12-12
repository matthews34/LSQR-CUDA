#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cusparse.h>

// #include "common.h"
#include "matrix.h"


bsrGPUMatrix bsrFromCsr(const cusparseHandle_t &handle,	const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const csrGPUMatrix &csrMatrix,
	const int m, const int n, const int blockDim)
{
	bsrGPUMatrix bsrMatrix;
	
	int base, nnzb;
	int mb = (m + blockDim - 1) / blockDim;
	int nb = (n + blockDim - 1) / blockDim;
	// allocate rowPtr
	cudaMalloc((void**)&bsrMatrix.bsrRowPtr, sizeof(int) * (mb + 1));
	// nnzTotalDevHostPtr points to host memory
    int* nnzTotalDevHostPtr = &nnzb;
	// compute number of nonzero blocks
	cusparseXcsr2bsrNnz(handle, dir, m, n,
        descr, csrMatrix.csrRowPtr, csrMatrix.csrColInd,
        blockDim,
        descr, bsrMatrix.bsrRowPtr,
        nnzTotalDevHostPtr);
	if (NULL != nnzTotalDevHostPtr) {
        nnzb = *nnzTotalDevHostPtr;
    } else {
        cudaMemcpy(&nnzb, bsrMatrix.bsrRowPtr + bsrMatrix.mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, bsrMatrix.bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }
	// allocate column indexes and values array
	cudaMalloc((void**)&bsrMatrix.bsrColInd, sizeof(int) * nnzb);
    cudaMalloc((void**)&bsrMatrix.bsrVal, sizeof(int) * (blockDim*blockDim) * nnzb);
	// convert matrix
	cusparseScsr2bsr(handle, dir, m, n,
        descr,
        csrMatrix.csrVal, csrMatrix.csrRowPtr, csrMatrix.csrColInd,
        blockDim,
        descr,
        bsrMatrix.bsrVal, bsrMatrix.bsrRowPtr, bsrMatrix.bsrColInd);

	bsrMatrix.nnzb = nnzb;
	bsrMatrix.blockDim = blockDim;
	bsrMatrix.mb = mb;
	bsrMatrix.nb = nb;

	return bsrMatrix;
}
// untested
void bsr2csr(const cusparseHandle_t &handle, const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const bsrGPUMatrix &bsrMatrix, csrGPUMatrix &csrMatrix,
	const int m, const int n)
{
	int mb = bsrMatrix.mb, nb = bsrMatrix.nb, blockDim = bsrMatrix.blockDim;
	cusparseSbsr2csr(handle, dir, mb, nb,
		descr, bsrMatrix.bsrVal, bsrMatrix.bsrRowPtr, bsrMatrix.bsrColInd, blockDim,
		descr, csrMatrix.csrVal, csrMatrix.csrRowPtr, csrMatrix.csrColInd);
}

void mvOperation(const cusparseHandle_t &handle, const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const float *alpha, const bsrGPUMatrix bsrMatrix, const float *x, const float *beta, float *y)
{
	cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE,
        bsrMatrix.mb, bsrMatrix.nb, bsrMatrix.nnzb, 
        alpha, descr, bsrMatrix.bsrVal, bsrMatrix.bsrRowPtr, bsrMatrix.bsrColInd, bsrMatrix.blockDim,
        x, beta, y);	
}

csrCPUMatrix matrix_alloc_cpu(int m, int nnz)
{
	csrCPUMatrix matrix;
	matrix.nnz = nnz;
	matrix.csrVal = new float[nnz];
	matrix.csrRowPtr = new int[m + 1];
	matrix.csrColInd = new int[nnz];
	return matrix;
}
void matrix_free_cpu(csrCPUMatrix &matrix)
{
	delete[] matrix.csrVal;
	delete[] matrix.csrRowPtr;
	delete[] matrix.csrColInd;
}

csrGPUMatrix matrix_alloc_gpu(int m, int nnz)
{
	csrGPUMatrix matrix;
	matrix.nnz = nnz;

	cudaMalloc((void**)&matrix.csrVal, sizeof(float) * nnz);
	cudaMalloc((void**)&matrix.csrRowPtr, sizeof(int) * (m + 1));
	cudaMalloc((void**)&matrix.csrColInd, sizeof(int) * nnz);

	return matrix;
}
void matrix_free_gpu_csr(csrGPUMatrix &matrix)
{
	cudaFree(matrix.csrVal);
	cudaFree(matrix.csrRowPtr);
	cudaFree(matrix.csrColInd);
}
void matrix_free_gpu_bsr(bsrGPUMatrix &matrix)
{
	cudaFree(matrix.bsrVal);
	cudaFree(matrix.bsrRowPtr);
	cudaFree(matrix.bsrColInd);
}

void matrix_upload(const csrCPUMatrix &src, csrGPUMatrix &dst, int m, int nnz)
{
	if (cudaMemcpy(dst.csrVal, src.csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Float\n");
	}
	if (cudaMemcpy(dst.csrRowPtr, src.csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Int1\n");
	}
	if (cudaMemcpy(dst.csrColInd, src.csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Int2\n");
	}
}
void matrix_download(const csrGPUMatrix &src, csrCPUMatrix &dst, int m, int nnz)
{
	cudaMemcpy(&dst.csrVal, &src.csrVal, sizeof(float) * nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(&dst.csrRowPtr, &src.csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dst.csrColInd, &src.csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
}
