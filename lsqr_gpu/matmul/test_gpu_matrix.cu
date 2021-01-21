#include <iostream>
#include "GPUMatrix.h"
#include "GPUVector.h"
#include "SpMat.h"

#include "utils.h"

cudaError_t cudaStat = cudaSuccess;

int main()
{
	// m  is pitch of the matrix resulting in B_t as default
    int m = 7, n = 6;
	// initialize values
	double B_t[] = { 10.0, 0.0, 0.0, 0.0, -2.0, 0.0, 
					3.0, 9.0, 0.0, 0.0, 0.0, 3.0, 
					0.0, 7.0, 8.0, 7.0, 0.0, 0.0, 
					3.0, 9.0, 0.0, 0.0, 0.0, 3.0, 
					0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
					0.0, 8.0, 0.0, 9.0, 9.0, 13.0,  
					0.0, 4.0, 0.0, 0.0, 2.0, -1.0};
	double x0[] = {1.0, 2.0, 3.0, 4.0, 5.0, 7.0,9,10}; 
	double x1[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; 
	
	//create GPUVectors
	GPUVector x(n,x1);
	GPUVector x2(m,x1);
	//constructor without data creates array with m zeros
	GPUVector y(m);
	GPUVector y2(n);
	
	// calculate dot
	SpMat mat(m,n,B_t);
	mat.dot(x,y);
	
	double *y_cpu = new double[m];
	cudaMemcpy(y_cpu, y.elements, sizeof(double) * m, cudaMemcpyDeviceToHost);
	std::cout << "B*x = (";
	for(int i = 0; i < m-1; i++)
		std::cout << y_cpu[i] << " ";
	std::cout << y_cpu[m-1] << ")" << std::endl;
	SpMat mat_t = mat.transpose();
	mat_t.dot(x2,y2);
	cudaMemcpy(y_cpu, y2.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);
	std::cout << "B'*x = (";
	for(int i = 0; i < n-1; i++)
		std::cout << y_cpu[i] << " ";
	std::cout << y_cpu[n-1] << ")" << std::endl;
	GPUVector y3 = mat_t*x2;
	cudaMemcpy(y_cpu, y3.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);
	std::cout << "B'*x = (";
	for(int i = 0; i < n-1; i++)
		std::cout << y_cpu[i] << " ";
	std::cout << y_cpu[n-1] << ")" << std::endl;

	// calculate norm of x
	std::cout << "Calculating norm of x\n";
	std::cout << "||x|| = " << x.norm() << std::endl;

	// scale x
	std::cout << "Calculating x*2.5\n";
	double *h_v = new double[n];
	GPUVector d_v(n,x0);
	d_v = x * 2.5;
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x*2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	//calculate addition
	std::cout << "Calculating x + x*2.5\n";
	d_v = x + d_v;
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x + x*2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	//calculate subtraction
	std::cout << "Calculating x - x*2.5\n";
	d_v = 2.5 * x;
	d_v = x - d_v;
	cudaMemcpy(h_v, d_v.elements, sizeof(double) * n, cudaMemcpyDeviceToHost);	
	std::cout << "x - x * 2.5 = (";
	for(int i = 0; i < n-1; i++)
		std::cout << h_v[i] << " ";
	std::cout << h_v[n-1] << ")" << std::endl;

	FREE(y_cpu);
	FREE(h_v);
    cudaDeviceReset();
}