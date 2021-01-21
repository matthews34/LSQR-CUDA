#pragma once

#include "GPUVector.h"

#define BULK_SIZE 5

class SpMat {
public:
	int rows, cols, nnz;
	int *rowPtr;
	int *colInd;
	double* val;
	SpMat(int, int, double*);
	SpMat(	int* rowP, int* colI, double* values, 
			int rows, int cols, int nnz) : 
			rows(rows), cols(cols), nnz(nnz) {
				cudaMalloc(&rowPtr, (rows + 1) * sizeof(int));
				cudaMalloc(&colInd, (nnz) * sizeof(int));
				cudaMalloc(&val, (nnz) * sizeof(double));
				cudaMemcpy(rowPtr,rowP,sizeof(double)*(rows+1),cudaMemcpyDefault);
				cudaMemcpy(colInd,colI,sizeof(double)*nnz,cudaMemcpyDefault);
				cudaMemcpy(val,values,sizeof(double)*nnz,cudaMemcpyDefault);
			}
	SpMat(int rows,int cols,int nnz) : rows(rows), cols(cols), nnz(nnz) {
		cudaMalloc(&rowPtr,(rows+1)*sizeof(int));
		cudaMalloc(&colInd,nnz*sizeof(int));
		cudaMalloc(&val,nnz*sizeof(double));
	}
	void dot(const GPUVector&, GPUVector&);
	SpMat transpose();
	~SpMat();
	GPUVector operator*(const GPUVector b);
};