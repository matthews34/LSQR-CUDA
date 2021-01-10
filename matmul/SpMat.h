#pragma once

#include "GPUVector.h"

#define BULK_SIZE 2

class SpMat {
public:
	int rows, cols, nnz;
	int *rowPtr;
	int *colInd;
	double* val;
	SpMat(int, int, double*);
	SpMat(SpMat&);
	SpMat(	int* rowPtr, int* colInd, double* val, 
			int rows, int cols, int nnz) : 
			rowPtr(rowPtr), colInd(colInd), val(val),
			rows(rows), cols(cols), nnz(nnz) {}
	void dot(GPUVector&, GPUVector&);
	~SpMat();
};