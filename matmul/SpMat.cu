#include "SpMat.h"
#include "utils.h"
#include "stdio.h"

__global__ void nnz_in_row(const double* data_partial, const int n, const int cols, int* nnz) {
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(n < i)
		return;
	if(abs(data_partial[i]) > ZERO)
		atomicAdd(nnz, 1);
}

__global__ void cum_sum(const int * rowNnz, const int rows, int * cumsum) {
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(rows < idx)
		return;
	if(idx == rows-1)
		cumsum[0] = 0;
	else
		cumsum[idx+1] = rowNnz[idx];
	for(int stride = 1; stride < rows; stride*=2) {
		__syncthreads();
		if(stride < idx)
			cumsum[idx] = cumsum[idx] + cumsum[idx-stride];
	}
	__syncthreads();
	if(idx == rows)
		cumsum[idx] = cumsum[idx] + rowNnz[idx-1];
}

__global__ void get_ind_val(const double* data_partial, const int n, const int cols, int * colInd, double * val, int& nnz) {
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	//int elem_num = nnz;
	if(n < i)
		return;
	if(!(abs(data_partial[i]) > ZERO))
		return;
	int my_ind;
	my_ind = atomicSub(&nnz, 1) - 1;
	colInd[my_ind] = i;
	val[my_ind] = data_partial[i];
}

SpMat::SpMat(int rows, int cols, double * data) : rows(rows), cols(cols) {
	//dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dimBlock.x*dimBlock.y / (cols*BULK_SIZE) + 1);
	double * data_partial;
	int *RowNonzero;
	int *nnz_elem;
	cudaMalloc(&nnz_elem, sizeof(int));
	cudaMalloc(&data_partial, cols * sizeof(double));
	cudaMalloc(&RowNonzero, rows*sizeof(int));
	for(int i = 0; i < rows; i++) {
		cudaMemset(nnz_elem,0,sizeof(int));
		cudaMemcpy(data_partial, data + cols * i, cols * sizeof(double), cudaMemcpyDefault);
		nnz_in_row<<<dimGrid, dimBlock>>>(data_partial, cols, cols, nnz_elem);
		cudaMemcpy((RowNonzero + i), nnz_elem, sizeof(int),cudaMemcpyDefault);
	}
	cudaMalloc(&rowPtr, (rows + 1) * sizeof(int));
	cum_sum<<<dimGrid, dimBlock>>>(RowNonzero, rows, rowPtr);
	cudaMemcpy(&nnz,rowPtr + rows,sizeof(int),cudaMemcpyDefault);
	cudaMalloc(&colInd, (nnz) * sizeof(int));
	cudaMalloc(&val, (nnz) * sizeof(double));
	printf("Matrix has %d Non-Zero Elements\n",nnz);
	for(int i = 0; i < rows; i++) {	
		int offset;
		cudaMemcpy(&offset, rowPtr + i, sizeof(int), cudaMemcpyDefault);
		cudaMemcpy(data_partial, data + cols*i, cols * sizeof(double), cudaMemcpyDefault);
		get_ind_val<<<dimGrid, dimBlock>>>(data_partial, cols, cols, colInd + offset, val + offset, RowNonzero[i]);
	}
	CUDAFREE(RowNonzero);
	CUDAFREE(data_partial);
}

__global__ void dot_kernel(	const int * rowPtr, const int * colInd, const double* val, 
							const double* x, double* y, int row_num, int col_num, double * y_nnz){
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(rowPtr[row_num] - 1 < idx)
		return;
	int row;
	for (int i = 0; i < row_num; i++)
		if((idx >= rowPtr[i] && idx < rowPtr[i+1]) && rowPtr[i] != rowPtr[i+1]) {
			row = i;
		}
	y_nnz[idx] = x[colInd[idx]]*val[idx];
	if(idx != rowPtr[row])
		return;
	int n = rowPtr[row+1] - rowPtr[row];
	for(int i = 0; i < n; i++)
		y[row] += y_nnz[idx+i];
}

void SpMat::dot(GPUVector & x,GPUVector & y ) {
	// create nnz threads
	assert(x.n == cols);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dimBlock.x*dimBlock.y/nnz + 1);
	double *y_nnz;
	cudaMalloc(&y_nnz, nnz*sizeof(double));
	dot_kernel<<<dimGrid, dimBlock>>>(rowPtr, colInd, val, x.elements, y.elements, rows, cols, y_nnz);
	CUDAFREE(y_nnz);
}

__global__ void transpose_row_nnz(const int * colInd, int cols, int nnz, int* colNnz, int* cumsum) {
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(nnz - 1 < idx)
		return;
	atomicAdd(colNnz + colInd[idx],1);
	if(cols < idx)
		return;
	if(idx == cols-1)
		cumsum[0] = 0;
	else
		cumsum[idx+1] = colNnz[idx];
	for(int stride = 1; stride < cols; stride*=2) {
		__syncthreads();
		if(stride < idx)
			cumsum[idx] = cumsum[idx] + cumsum[idx-stride];
	}
	__syncthreads();
	if(idx == cols)
		cumsum[idx] = cumsum[idx] + colNnz[idx-1];
}

__global__ void transpose_kernel(	const int* rowPtr, const int * colInd, const double* val, 
									int * rowInd, double* trans_val,
									int row_num, int col_num, int nnz, int* colNnz, int* colPtr) {
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(nnz - 1 < idx)
		return;
	int row;
	for (int i = 0; i < row_num; i++)
		if((idx >= rowPtr[i] && idx < rowPtr[i+1]) && rowPtr[i] != rowPtr[i+1]) {
			row = i;
		}
	int my_ind = atomicSub(colNnz + colInd[idx], 1) - 1;
	rowInd[colPtr[colInd[idx]] + my_ind] = row;
	trans_val[colPtr[colInd[idx]] + my_ind] = val[idx];
}

SpMat::SpMat(SpMat &A) : rows(A.cols), cols(A.rows), nnz(A.nnz) {
	cudaMalloc(&rowPtr,(rows+1)*sizeof(int));
	cudaMalloc(&colInd,nnz*sizeof(int));
	cudaMalloc(&val,nnz*sizeof(double));
	int *rowNnz;
	cudaMalloc(&rowNnz,rows*sizeof(int));
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dimBlock.x*dimBlock.y/nnz + 1);
	transpose_row_nnz<<<dimGrid, dimBlock>>>(A.colInd, A.cols, nnz, rowNnz, rowPtr);
	transpose_kernel<<<dimGrid, dimBlock>>>(A.rowPtr, A.colInd, A.val, colInd, val, A.rows, A.cols, A.nnz, rowNnz, rowPtr);
	CUDAFREE(rowNnz);
}

SpMat::~SpMat() {
	CUDAFREE(rowPtr);
	CUDAFREE(colInd);
	CUDAFREE(val);
}