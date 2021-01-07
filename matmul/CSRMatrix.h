#pragma once

#include <iostream>
#include "utils.h"

// CSR and CSC do not differ in that context
// however CSR gives the transposed matrix as the default!!
struct CSRMatrix {
	// some descriptors and data
	int *csrRowPtr;
	int *csrColInd; 
	double *csrVal;
	int totalNnz;
	int M, N;
	cusparseMatDescr_t descr;
	cusparseHandle_t handle;
	// creates CSR Sparse matrix from GPUMatrix
	CSRMatrix(GPUMatrix A, cusparseHandle_t &cusparseHandle) : N(A.N), M(A.M) , handle(cusparseHandle){
		cusparseStat = cusparseCreateMatDescr(&descr);
		CUSPARSESUCCESS(cusparseStat);
		cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL );
		
		// get non zero element count per row
		// and where each row starts as ptr
		int *RowNonzero;
		cudaMallocManaged(&RowNonzero, M*sizeof(int));
		cudaMallocManaged(&csrRowPtr, (M+1)*sizeof(int));
		cusparseStat = cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, A.elements, M, RowNonzero, &totalNnz);
		CUSPARSESUCCESS(cusparseStat);
		
		// get index and value of elements in each row
		cudaMallocManaged(&csrColInd, totalNnz*sizeof(int));
		cudaMallocManaged(&csrVal, totalNnz*sizeof(double));
		cusparseStat = cusparseDdense2csr(handle, M, N, descr, A.elements, M, RowNonzero, csrVal, csrRowPtr, csrColInd); 
		CUSPARSESUCCESS(cusparseStat);
		
		// sync (maybe unneccessary)
		cudaDeviceSynchronize();
		CUDAFREE(RowNonzero);
	}
	// load all precalculated values into a CSRMatrix (needed for transposing the matrix)
	CSRMatrix(int *rowPtr, int *colInd, double *val, int nnz, int m, int n, cusparseHandle_t &cusparseHandle) : csrRowPtr(rowPtr), csrColInd(colInd), csrVal(val), totalNnz(nnz), N(n), M(m) , handle(cusparseHandle) {
		cusparseStat = cusparseCreateMatDescr(&descr);
		CUSPARSESUCCESS(cusparseStat);
		cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
		cudaDeviceSynchronize();
	}
	// calculates a dot product between "this" and x and writes it to y
	void dot(GPUVector &x, GPUVector &y) {
		double h_one = 1.0;
		double h_zero = 0.0;
		size_t buffer_size = 0;
		// calculate the required space for the calculation
		cusparseStat = cusparseCsrmvEx_bufferSize(handle,
 					CUSPARSE_ALG_MERGE_PATH,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    M,
                    M,
                    totalNnz,
                    &h_one, CUDA_R_64F,
                    descr,
                    csrVal, CUDA_R_64F,
                    csrRowPtr,
                    csrColInd,
                    x.elements, CUDA_R_64F,
                    &h_zero, CUDA_R_64F,
                    y.elements, CUDA_R_64F,
					CUDA_R_64F,
					&buffer_size);
		CUSPARSESUCCESS(cusparseStat);
		// load the buffer
		void* buffer = NULL;
		cudaStat = cudaMalloc ((void**)&buffer, buffer_size);
		CUDASUCCESS(cudaStat);
		// execute the calculation inside the buffer
        cusparseStat = cusparseCsrmvEx(handle,
					CUSPARSE_ALG_MERGE_PATH,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    M,
                    M,
                    totalNnz,
                    &h_one, CUDA_R_64F,
                    descr,
                    csrVal, CUDA_R_64F,
                    csrRowPtr,
                    csrColInd,
                    x.elements, CUDA_R_64F,
                    &h_zero, CUDA_R_64F,
                    y.elements, CUDA_R_64F,
					CUDA_R_64F,
			        buffer);
		CUSPARSESUCCESS(cusparseStat);
		CUDAFREE(buffer);
	}
	// returns a transposed CSRMatrix (basically a CSC Matrix.. it is confusing)
	CSRMatrix transpose() {
		size_t buffer_size = 0;
		// preallocate all required pointers
        int* cscColPtr = NULL;
        int* cscRowInd = NULL;
        double* cscVal = NULL;
		cudaMallocManaged(&cscColPtr, (N+1)*sizeof(int));
		cudaMallocManaged(&cscRowInd, totalNnz*sizeof(int));
		cudaMallocManaged(&cscVal, totalNnz*sizeof(double));
		// calculate the required space
		cusparseStat = cusparseCsr2cscEx2_bufferSize(handle,
                              M,
                              N,
                              totalNnz,
                              csrVal,
                              csrRowPtr,
                              csrColInd,
                              cscVal,
                              cscColPtr,
                              cscRowInd,
                              CUDA_R_64F,
                              CUSPARSE_ACTION_NUMERIC,
                              CUSPARSE_INDEX_BASE_ZERO,
                              CUSPARSE_CSR2CSC_ALG1,
                              &buffer_size);
		CUSPARSESUCCESS(cusparseStat);
		// load the buffer
		void* buffer = NULL;
		cudaStat = cudaMalloc ((void**)&buffer, buffer_size);
		CUDASUCCESS(cudaStat);
		// execute transposition in the buffer
		cusparseStat = cusparseCsr2cscEx2(handle,
                   M,
                   N,
                   totalNnz,
                   csrVal,
                   csrRowPtr,
                   csrColInd,
                   cscVal,
                   cscColPtr,
                   cscRowInd,
                   CUDA_R_64F,
                   CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO,
                   CUSPARSE_CSR2CSC_ALG1,
                   buffer);
		CUSPARSESUCCESS(cusparseStat);
		//create new matrix with the pointers and return it
		CSRMatrix mat(cscColPtr,cscRowInd,cscVal,totalNnz,M,N,handle);
		return mat;
	}
	// some destructors
	~CSRMatrix() {
		CUDAFREE(csrRowPtr);
		CUDAFREE(csrColInd);
		CUDAFREE(csrVal);
		CUSPARSEFREEMATRIX(descr);
	}
};
