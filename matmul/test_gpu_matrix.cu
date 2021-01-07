#include <iostream>
#include "GPUMatrix.h"
#include "GPUVector.h"
#include "CSRMatrix.h"
#include "utils.h"

cudaError_t cudaStat = cudaSuccess;
cusparseStatus_t cusparseStat = CUSPARSE_STATUS_SUCCESS;

int main()
{
	// m  is pitch of the matrix resulting in B_t as default
    int m = 6, n = 6;
	// initialize values
	double B_t[] = { 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 
					3.0, 9.0, 0.0, 0.0, 0.0, 3.0, 
					0.0, 7.0, 8.0, 7.0, 0.0, 0.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 8.0, 0.0, 9.0, 9.0, 13.0,  
					0.0, 4.0, 0.0, 0.0, 2.0, -1.0};
	double x0[] = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0,9,10}; 
	double *y0 = new double[m];	
	
	//set cusparse handles
	cusparseHandle_t cusparseHandle;
	cusparseCreate(&cusparseHandle);
	
	//create GPUMatrix and CSRMatrix from it
	GPUMatrix B_t_gpu(n,m,B_t);
	CSRMatrix B_t_csr(B_t_gpu,cusparseHandle);
	
	//create GPUVectors
	GPUVector x(n,x0);
	GPUVector y(m,y0);
	
	// get transpose matrix of B_t which is B
	CSRMatrix B_csr = B_t_csr.transpose();

	// calculate norm of x
	std::cout << "Calculating norm of x\n";
	std::cout << "||x|| = " << x.norm() << std::endl;

	// scale x
	std::cout << "Calculating x*2.5\n";
	double *h_v = new double[n];
	GPUVector d_v(n,x0);
	x.scale(d_v, 2.5);
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x*2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	//calculate addition
	std::cout << "Calculating x + x*2.5\n";
	x.add(d_v, d_v);
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x + x*2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	//calculate subtraction
	std::cout << "Calculating x - x*2.5\n";
	x.scale(d_v, 2.5);
	x.sub(d_v, d_v);
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x - x * 2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	//calculate dot product and write it to y
	std::cout << "Calculating B*x\n";
	B_csr.dot(x, y);

	//create y_cpu for loading the results from gpu
	double *y_cpu = new double[m];
	from_gpu(y_cpu,y.elements,m);
	std::cout << "B*x = (";
	for(int i = 0; i < m-1; i++)
		std::cout << y_cpu[i] << " ";
	std::cout << y_cpu[m-1] << ")" << std::endl;

	//calculate dot product and write it to y
	std::cout << "Calculating B'*x\n";
	B_t_csr.dot(x, y);
	
	from_gpu(y_cpu,y.elements,m);
	std::cout << "B'*x = (";
	for(int i = 0; i < m-1; i++)
		std::cout << y_cpu[i] << " ";
	std::cout << y_cpu[m-1] << ")" << std::endl;

	FREE(y_cpu);
	FREE(y0);
	CUSPARSEFREEHANDLE(cusparseHandle);
    cudaDeviceReset();
}