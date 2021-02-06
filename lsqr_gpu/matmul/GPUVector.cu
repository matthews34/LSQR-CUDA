#include "GPUVector.h"
#include "utils.h"
#include <cublas_v2.h>

// kernel for norm calculation (DEPRICATED)
__global__ void norm_kernel(const double *data, const int n, double *output) {

	__shared__ double partialSum[2*BLOCK_SIZE];

	int i = threadIdx.x;

	int start = 2 * blockIdx.x * blockDim.x;

	// Each thread loads two elements
	partialSum[i] = data[start + i] * data[start + i];
	partialSum[blockDim.x + i] = data[start + blockDim.x + i] * data[start + blockDim.x + i];

	for (int stride = blockDim.x; stride > 0; stride /= 2) {
		__syncthreads();
		if (i < stride) partialSum[i] += partialSum[i + stride];
	}

	if (i == 0) *output = sqrt(partialSum[0]);
}

// determines norm using cublas
double GPUVector::norm() {
	double result;

	cublasStatus_t status;
	status = cublasDnrm2(handle, n, elements, 1, &result);
	if (status != CUBLAS_STATUS_SUCCESS) {
		printf("Error calculating norm\n");
		exit(-1);
	}

	return result;
}

// kernel for scaling vector
__global__ void scale_kernel(const double *input, double *output, const int n, const double s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(n - 1 < i)
		return;
	output[i] = input[i] * s;
}

// star operator for scaling vector
GPUVector operator*(const GPUVector v, const double s) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((v.n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(v.handle, v.n);
	
	scale_kernel<<<dimGrid, dimBlock>>>(v.elements, c.elements, v.n, s);

	return c;
}
// star operator for scaling vector
GPUVector operator*(const double s, const GPUVector v) {
	return v * s;
}

// kernel for adding vectors a, b and storing results in c
__global__ void add_kernel(const double *a, const double *b, double *c, double factor) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	c[i] = a[i] + b[i] * factor;
}

//computes addition of current vector with b and stores result in c
GPUVector GPUVector::operator+(const GPUVector b) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(handle, n);
	
	add_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, c.elements, 1);

	return c;
}

//computes subraction of current vector and b and stores result in c
GPUVector GPUVector::operator-(const GPUVector b) {
	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	GPUVector c(handle, n);

	add_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, c.elements, -1);

	return c;
}