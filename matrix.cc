#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

#include <cuda_runtime.h>
#include <cusparse.h>

#include "common.h"
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

void bsrmv(cusparseHandle_t &handle, cusparseDirection_t &dir, cusparseMatDescr_t &descr,
	const float *alpha, const bsrGPUMatrix bsrMatrix, const float *x, const float *beta, float *y)
{
	cusparseSbsrmv(handle, dir, CUSPARSE_OPERATION_NON_TRANSPOSE,
        bsrMatrix.mb, bsrMatrix.nb, bsrMatrix.nnzb, 
        alpha, descr, bsrMatrix.bsrVal, bsrMatrix.bsrRowPtr, bsrMatrix.bsrColInd, bsrMatrix.blockDim,
        x, beta, y);	
}
void csrmv(cusparseHandle_t &handle, const cusparseMatDescr_t &descr,
	int m, int n, const float *alpha, const csrGPUMatrix matrix, const float *x, const float *beta, float *y)
{
	// Get buffer size
	size_t bufferSizeInBytes;
	cusparseCsrmvEx_bufferSize(handle, CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
		m, n, matrix.nnz, alpha, CUDA_R_32F, descr,
		matrix.csrVal, CUDA_R_32F, matrix.csrRowPtr, matrix.csrColInd, 
		x, CUDA_R_32F, beta, CUDA_R_32F, y, CUDA_R_32F, CUDA_R_32F, &bufferSizeInBytes);
	void* buffer;
	// Allocate buffer
	cudaMalloc((void**)&buffer, bufferSizeInBytes);
	// Compute operation
	cusparseCsrmvEx(handle, CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
		m, n, matrix.nnz, alpha, CUDA_R_32F, descr,
		matrix.csrVal, CUDA_R_32F, matrix.csrRowPtr, matrix.csrColInd,
		x, CUDA_R_32F, beta, CUDA_R_32F, y, CUDA_R_32F, CUDA_R_32F, buffer);
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
	cudaMemcpy(dst.csrVal, src.csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);
	cudaMemcpy(dst.csrRowPtr, src.csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(dst.csrColInd, src.csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
}
void matrix_download(const csrGPUMatrix &src, csrCPUMatrix &dst, int m, int nnz)
{
	cudaMemcpy(&dst.csrVal, &src.csrVal, sizeof(float) * nnz, cudaMemcpyDeviceToHost);
	cudaMemcpy(&dst.csrRowPtr, &src.csrRowPtr, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(&dst.csrColInd, &src.csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
}

float* randomVector(int n)
{
	float *vector;
	vector = (float*)malloc(n * sizeof(float));
	
	srand(1);
	for (int i = 0; i < n; i++)
	{
		vector[i] = -100 + static_cast <float> (rand()) / static_cast <float> (RAND_MAX/200);
	}
	
	return vector;
}
csrCPUMatrix matrixFromCsv(cusparseHandle_t handle, std::string filename, int m, int n)
{
	csrCPUMatrix matrix;
	std::vector<float> values;
	std::vector<int> rowPtr;
	std::vector<int> colInd;

	// Create an input filestream
    std::ifstream file(filename);

	// Make sure the file is open
    if(!file.is_open()) throw std::runtime_error("Could not open file");

	// Helper vars
    std::string line, colname;

	// Read the column names
    if(file.good())
    {
        // Extract the first line in the file
        std::getline(file, line);

        // Create a stringstream from line
        std::stringstream ss(line);

        // Extract each column name
        while(std::getline(ss, colname, ',')){
			// Do nothing
        }
    }
	
	int rowId = 0;
	while (std::getline(file, line))
	{
		int row, col;
		float val;

		std::stringstream ss(line);

		// Get rowPtr
		if (rowId < m + 1)
		{
			ss >> row;
			rowPtr.push_back(row);	
			if (ss.peek() == ',') ss.ignore();
		} else 
		{
			char buff;
			ss >> buff;
			if (ss.peek() == ',') ss.ignore();
		}

		// Get colInd
		ss >> col;
		colInd.push_back(col);
		if (ss.peek() == ',') ss.ignore();

		// Get value
		ss >> val;
		values.push_back(val);
		if (ss.peek() == ',') ss.ignore();

		rowId++;
	}
	
    // Close file
    file.close();

	int nnz = values.size();
	matrix = matrix_alloc_cpu(m, nnz);
	std::copy(colInd.begin(), colInd.end(), matrix.csrColInd);
	std::copy(values.begin(), values.end(), matrix.csrVal);
	std::copy(rowPtr.begin(), rowPtr.end(), matrix.csrRowPtr);
	
	return matrix;
}