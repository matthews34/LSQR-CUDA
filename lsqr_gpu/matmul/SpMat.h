#pragma once

#include "GPUVector.h"
#include <stdio.h>
#include <cuda.h>
#include <cusparse.h>


#define BULK_SIZE 5

// header with data for SpMat
class SpMat {
public:
	int rows, cols, nnz;
	int *rowPtr;
	int *colInd;
	double* val;
	cusparseHandle_t cusparseH;
	cusparseMatDescr_t descrA;
	cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;
	SpMat(int, int, double*);
	// constructor for precalculated values;
	SpMat(	int* rowP, int* colI, double* values, 
			int rows, int cols, int nnz, cusparseHandle_t cusparseH) : 
			rows(rows), cols(cols), nnz(nnz), cusparseH(cusparseH) {
				// printf("Initializing Matrix\n");
				cudaMalloc(&rowPtr, (rows + 1) * sizeof(int));
				cudaMalloc(&colInd, (nnz) * sizeof(int));
				cudaMalloc(&val, (nnz) * sizeof(double));
				cudaMemcpy(rowPtr,rowP,sizeof(int)*(rows+1),cudaMemcpyHostToDevice);
				cudaMemcpy(colInd,colI,sizeof(int)*nnz,cudaMemcpyHostToDevice);
				cudaMemcpy(val,values,sizeof(double)*nnz,cudaMemcpyHostToDevice);
				cusparseStat = cusparseCreateMatDescr(&descrA);
				assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
			}
	SpMat(int rows,int cols,int nnz) : rows(rows), cols(cols), nnz(nnz) {
		cudaMalloc(&rowPtr,(rows+1)*sizeof(int));
		cudaMalloc(&colInd,nnz*sizeof(int));
		cudaMalloc(&val,nnz*sizeof(double));
	}
	void dot(const GPUVector&, GPUVector&);
	SpMat transpose();
	~SpMat();
	GPUVector operator*(const GPUVector & b);
};