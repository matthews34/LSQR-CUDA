#include "SpMat.h"
#include "utils.h"
#include "stdio.h"

//method for calculating number of non-zero elements in a row (DEPRICATED)
__global__ void nnz_in_row(const double* data_partial, const int n, const int cols, int* nnz) {
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(n-1 < i)
		return;
	if(abs(data_partial[i]) > ZERO)
		atomicAdd(nnz, 1);
}

//method for computing the cummulative sum of non-zero elements along the rows (DEPRICATED)
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

//method for extracting column index and according value from each row (DEPRICATED)
__global__ void get_ind_val(const double* data_partial, const int n, const int cols, int * colInd, double * val, int& nnz) {
	int i = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(n-1 < i)
		return;
	if(!(abs(data_partial[i]) > ZERO))
		return;
	int my_ind;
	my_ind = atomicSub(&nnz, 1) - 1;
	colInd[my_ind] = i;
	val[my_ind] = data_partial[i];
	
}

// This constructor is DEPRICATED
// it basically loads the data batchwise to the GPU and computes the CSR representation of the Matrix
SpMat::SpMat(int rows, int cols, double * data) : rows(rows), cols(cols) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(dimBlock.x*dimBlock.y / (cols*BULK_SIZE) + 1);
	double * data_partial;
	int *RowNonzero;
	int *nnz_elem;
	cudaMalloc(&nnz_elem, sizeof(int));
	cudaMalloc(&data_partial, cols * BULK_SIZE * sizeof(double));
	cudaMalloc(&RowNonzero, rows*sizeof(int));
	int i_it_num = rows % BULK_SIZE == 0 ? rows/BULK_SIZE : rows/BULK_SIZE + 1;
	int j_it_num, elem_num;
	for(int i = 0; i < i_it_num; i++) {		
		elem_num = (BULK_SIZE*cols) * (i+1) < rows*cols ? (BULK_SIZE*cols) : rows*cols - (BULK_SIZE*cols)*(i);
		cudaMemcpy(data_partial, data + (BULK_SIZE*cols) * i, elem_num * sizeof(double), cudaMemcpyDefault);
		j_it_num = BULK_SIZE*(i+1) < rows ? BULK_SIZE : rows - BULK_SIZE*i;
		for (int j = 0; j < j_it_num; j++)
			nnz_in_row<<<dimGrid, dimBlock>>>(data_partial + cols*j, cols, cols, (RowNonzero + j) + BULK_SIZE*i);
	}
	cudaMalloc(&rowPtr, (rows + 1) * sizeof(int));
	cum_sum<<<dimGrid, dimBlock>>>(RowNonzero, rows, rowPtr);
	cudaMemcpy(&nnz,rowPtr + rows,sizeof(int),cudaMemcpyDefault);
	cudaMalloc(&colInd, (nnz) * sizeof(int));
	cudaMalloc(&val, (nnz) * sizeof(double));
	printf("Matrix has %d Non-Zero Elements\n",nnz);
		
	int offset, row_num;	
	for(int i = 0; i < i_it_num; i++) {	
		elem_num = (BULK_SIZE*cols) * (i+1) < rows*cols ? (BULK_SIZE*cols) : rows*cols - (BULK_SIZE*cols)*(i);
		cudaMemcpy(data_partial, data + (BULK_SIZE*cols) * i, elem_num * sizeof(double), cudaMemcpyDefault);
		j_it_num = BULK_SIZE*(i+1) < rows ? BULK_SIZE : rows - BULK_SIZE*i;
		for (int j = 0; j < j_it_num; j++) {
			row_num = j + BULK_SIZE*i;
			cudaMemcpy(&offset, rowPtr + row_num, sizeof(int), cudaMemcpyDefault);
			get_ind_val<<<dimGrid, dimBlock>>>(data_partial + cols*j, cols, cols, colInd + offset, val + offset, RowNonzero[row_num]);
		}
	}
	CUDAFREE(RowNonzero);
	CUDAFREE(data_partial);
}

//kernel for calculating the dot product between x and A and storing the results in y (DEPRICATED)
__global__ void dot_kernel(	const int * rowPtr, const int * colInd, const double* val, 
							const double* x, double* y, int row_num, int col_num, double * y_nnz){
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(rowPtr[row_num] - 1 < idx)
		return;
	// each thread has to determine its row number (which was a bad idea)
	int row;
	for (int i = 0; i < row_num; i++)
		if((idx >= rowPtr[i] && idx < rowPtr[i+1]) && rowPtr[i] != rowPtr[i+1]) {
			row = i;
		}
	// each thread has to calculate its value
	y_nnz[idx] = x[colInd[idx]]*val[idx];
	if(idx != rowPtr[row])
		return;
	//first thread of a row sums all entrys in the row and stores result in the according entry in y
	int n = rowPtr[row+1] - rowPtr[row];
	for(int i = 0; i < n; i++)
		y[row] += y_nnz[idx+i];
}

// method that calculates dot product between this matrix and GPUVector x and stores results to GPUVector y
void SpMat::dot(const GPUVector & x,GPUVector & y ) {
	assert(x.n == cols);
	size_t buffer_size;
	double h_one = 1.0;
    const double h_zero = 0.0;
	// calculate buffer size
	cusparseStat = cusparseCsrmvEx_bufferSize(cusparseH,
					 CUSPARSE_ALG_MERGE_PATH,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     rows,
                     cols,
                     nnz,
                     &h_one, CUDA_R_64F,
                     descrA,
                     val, CUDA_R_64F,
                     rowPtr,
                     colInd,
                     x.elements, CUDA_R_64F,
                     &h_zero, CUDA_R_64F,
                     y.elements, CUDA_R_64F,
					 CUDA_R_64F,
					 &buffer_size);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
	// allocate buffer for calculation
	void* buffer;
	cudaMalloc ((void**)&buffer, buffer_size);
	cusparseStat = cusparseCsrmvEx(cusparseH,
					 CUSPARSE_ALG_MERGE_PATH,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     rows,
                     cols,
                     nnz,
                     &h_one, CUDA_R_64F,
                     descrA,
                     val, CUDA_R_64F,
                     rowPtr,
                     colInd,
                     x.elements, CUDA_R_64F,
                     &h_zero, CUDA_R_64F,
                     y.elements, CUDA_R_64F,
					 CUDA_R_64F,
			         buffer);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
	CUDAFREE(buffer);
}

// star operator for calling the dot product
GPUVector SpMat::operator*(const GPUVector &b) {
	GPUVector y(b.handle, rows);
	dot(b,y);
	return y;
}

// kernel for calculating the number of nnz elements in each column of matrix A
__global__ void transpose_row_nnz(const int * colInd, int cols, int nnz, int* colNnz, int* cumsum) {
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(nnz - 1 < idx)
		return;
	// increment number of ellements in your column
	atomicAdd(colNnz + colInd[idx],1);
	if(cols + 1 < idx)
		return;
	if(idx == cols-1)
		cumsum[0] = 0;
	else
		cumsum[idx+1] = colNnz[idx];
	// use predicate sum approach from lecture to compute cummulative sum
	for(int stride = 1; stride < cols; stride*=2) {
		__syncthreads();
		if(stride < idx)
			cumsum[idx] = cumsum[idx] + cumsum[idx-stride];
	}
	__syncthreads();
	if(idx == cols)
		cumsum[idx] = cumsum[idx] + colNnz[idx-1];
}

// kernel for transposing the entries in matrix A
__global__ void transpose_kernel(	const int* rowPtr, const int * colInd, const double* val, 
									int * rowInd, double* trans_val,
									int row_num, int col_num, int nnz, int* colNnz, const int* colPtr) {
	int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
	if(nnz < idx)
		return;
	// each thread has to find its row
	int row;
	for (int i = 0; i < row_num; i++)
		if((idx >= rowPtr[i] && idx < rowPtr[i+1]) && rowPtr[i] != rowPtr[i+1]) {
			row = i;
		}
	int my_ind = atomicSub(colNnz + colInd[idx], 1) - 1;
	rowInd[colPtr[colInd[idx]] + my_ind] = row;
	trans_val[colPtr[colInd[idx]] + my_ind] = val[idx];
}

SpMat SpMat::transpose() {
	size_t buffer_size = 0;
	// preallocate all required pointers
	int* cscColPtr = NULL;
	int* cscRowInd = NULL;
	double* cscVal = NULL;
	cudaMalloc(&cscColPtr, (cols+1)*sizeof(int));
	cudaMalloc(&cscRowInd, nnz*sizeof(int));
	cudaMalloc(&cscVal, nnz*sizeof(double));
	// calculate the required space
	cusparseStat = cusparseCsr2cscEx2_bufferSize(cusparseH,
						  rows,
						  cols,
						  nnz,
						  val,
						  rowPtr,
						  colInd,
						  cscVal,
						  cscColPtr,
						  cscRowInd,
						  CUDA_R_64F,
						  CUSPARSE_ACTION_NUMERIC,
						  CUSPARSE_INDEX_BASE_ZERO,
						  CUSPARSE_CSR2CSC_ALG1,
						  &buffer_size);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
	// load the buffer
	void* buffer = NULL;
	cudaMalloc ((void**)&buffer, buffer_size);
	// execute transposition in the buffer
	cusparseStat = cusparseCsr2cscEx2(cusparseH,
			   rows,
			   cols,
			   nnz,
			   val,
			   rowPtr,
			   colInd,
			   cscVal,
			   cscColPtr,
			   cscRowInd,
			   CUDA_R_64F,
			   CUSPARSE_ACTION_NUMERIC,
			   CUSPARSE_INDEX_BASE_ZERO,
			   CUSPARSE_CSR2CSC_ALG1,
			   buffer);
	assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
	//create new matrix with the pointers and return it
	SpMat mat(cscColPtr,cscRowInd,cscVal, cols, rows, nnz, cusparseH);
	CUDAFREE(buffer);
	return mat;
}
	
// destructor for constructed arrays
SpMat::~SpMat() {
	CUDAFREE(rowPtr);
	CUDAFREE(colInd);
	CUDAFREE(val);
}