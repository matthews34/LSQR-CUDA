#include "GPUVector.h"
#include "utils.h"


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

double GPUVector::norm() {
	double h_result;
	double *d_result;

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	cudaMalloc(&d_result, sizeof(double));

	norm_kernel<<<dimGrid, dimBlock>>>(elements, n, d_result);

	cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
	CUDAFREE(d_result);

	return h_result;
}

__global__ void scale_kernel(const double *input, double *output, const int n, const double s) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	output[i] = input[i] * s;
}

void GPUVector::scale(const double s, GPUVector v) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	scale_kernel<<<dimGrid, dimBlock>>>(elements, v.elements, n, s);
}

__global__ void add_kernel(const double *a, const double *b, double *c, double factor) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	c[i] = a[i] + b[i] * factor;
}

void GPUVector::add(const GPUVector b, GPUVector out) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	add_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, out.elements, 1);
}

void GPUVector::sub(const GPUVector b, GPUVector out) {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((n + dimBlock.x - 1) / dimBlock.x);

	add_kernel<<<dimGrid, dimBlock>>>(elements, b.elements, out.elements, -1);
}