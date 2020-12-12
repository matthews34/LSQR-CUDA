#pragma once

#include <cstdlib>

struct csrCPUMatrix {
	int nnz;		// number of nonzero elements
	float* csrVal;	// size = nnz
	int* csrRowPtr;	// size = m+1
	int* csrColInd;	// size = nnz
};
struct csrGPUMatrix {
	int nnz;		// number of nonzero elements
	float* csrVal;	// size = nnz
	int* csrRowPtr;	// size = m+1
	int* csrColInd;	// size = nnz
};
struct bsrGPUMatrix {
	int blockDim;	// block dimension
	int mb; 		// number of block rows
					// mb = (m+blockDim-1)/blockDim
	int nb; 		// number of block columns
					// nb = (n+blockDim-1)/blockDim
	int nnzb; 		// number of nonzero blocks
	float* bsrVal; 	// size = nnzb*blockDim²
	int* bsrRowPtr;	// size = mb + 1
	int* bsrColInd;	// size = nnzb
};

bsrGPUMatrix bsrFromCsr(const cusparseHandle_t &handle,	const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const csrGPUMatrix &csrMatrix,
	const int m, const int n, const int blockDim);

void bsr2csr(const cusparseHandle_t &handle, const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const bsrGPUMatrix &bsrMatrix, csrGPUMatrix &csrMatrix,
	const int m, const int n, const int blockDim);

void mvOperation(const cusparseHandle_t &handle, const cusparseDirection_t &dir, const cusparseMatDescr_t &descr,
	const float *alpha, const bsrGPUMatrix bsrMatrix, const float *x, const float *beta, float *y);

csrCPUMatrix matrix_alloc_cpu(int m, int nnz);
void matrix_free_cpu(csrCPUMatrix &matrix);

csrGPUMatrix matrix_alloc_gpu(int m, int nnz);
void matrix_free_gpu_csr(csrGPUMatrix &matrix);
void matrix_free_gpu_bsr(bsrGPUMatrix &matrix);

void matrix_upload(const csrCPUMatrix &src, csrGPUMatrix &dst, int m, int nnz);
void matrix_download(const csrGPUMatrix &src, csrCPUMatrix &dst, int m, int nnz);