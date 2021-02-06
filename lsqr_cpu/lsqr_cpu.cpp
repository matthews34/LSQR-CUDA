#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>

#define ZERO 0.00000001

template<typename Vec>
double norm(Vec &v) {
    return v.norm();
}

template<typename Mat>
Mat transpose(Mat &m) {
    return m.transpose();
}

template<typename Mat, typename Vec>
Vec dot(Mat &A, Vec& b) {
    return A*b;
}

template<typename Vec>
Vec scale(Vec& v, double s) {
    return v*s;
}

template<typename Vec>
int size(Vec& x) {
    return x.size();
}

template<typename Mat, typename Vec>
void lsqr(Mat& A, Vec& b, Vec& x) {
    // (1) Initialization
    double beta = norm(b);
    Vec u = scale(b,1/beta);
    Mat A_t = transpose(A);
    Vec v = dot(A_t,u);
    double alpha = norm(v);
    v = scale(v,1/alpha);
    Vec w = v;
    double phi_hat = beta;
    double rho_hat = alpha;
    // (2) Iteration
    int it_max = size(x);
    double epsilon = 0.001;
    double rho, phi, c, s, theta, residual;
    for(int i = 0; i < it_max; i++) {
        // (3) Bidiagonalization
        u = dot(A,v) - scale(u,alpha);
        beta = norm(u);
        u = scale(u,1/beta);
        v = dot(A_t,u) - scale(v,beta);
        alpha = norm(v);
        v = scale(v,1/alpha);
        // (4) Orthogonal Transformation
        rho = std::sqrt(rho_hat*rho_hat + beta*beta);
        c = rho_hat / rho;
        s = beta / rho;
        theta = s * alpha;
        rho_hat = -c * alpha;
        phi = c * phi_hat;
        phi_hat = s * phi_hat;
        // (5) Update x, w
        x = x + scale(w, phi / rho );
        w = v - scale(w, theta / rho);
        residual = norm(dot(A,x) - b);
	//	printf("iteration %d: residual=%f\n",i,residual);
        if(residual < epsilon) {
            printf("finished after %d iterations\n",i);
            return;
        }
    }
    printf("it_max exeeded\n");
}


// reads vector from file
Eigen::VectorXd read_vector(char* file_name, int &n) {
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	Eigen::VectorXd vec(n);
	printf("Vector size: %d\n",n);
	
	fread(vec.data(),sizeof(double),n,file);
	fclose(file);
	
	return vec;
}

// reads matrix from file and parses it to csr format
Eigen::SparseMatrix<double> read_sparse_matrix(char* file_name, int& n, int& m) {
	printf("file_name = %s\n",file_name);
	
	FILE *file = fopen(file_name, "rb");
	if (file==NULL) {fputs ("File error",stderr); exit (1);}
	
	char *token = strtok(file_name, "_");
	token = strtok(NULL, "_");
	m = std::stoi( token );
	token = strtok(NULL, "_");
	n = std::stoi( token );
	
	printf("Matrix size: %dx%d\n",m,n);
	
	Eigen::SparseMatrix<double> mat(m,n);
	mat.reserve(Eigen::VectorXi::Constant(m,n/10));
	
	double *data = (double*) malloc (sizeof(double) * n);
	
	if (data == NULL) {fputs ("Memory error",stderr); exit (2);}
	int j = 0;
	while(fread(data,sizeof(double),n,file)) {
		for(int i = 0; i < n; i++)
		if(std::abs(data[i]) > ZERO) {
			mat.coeffRef(j,i) += data[i];
		}
		++j;
	}
	mat.makeCompressed();
	
	fclose(file);
	printf("Read Data\n");
	if(data) free(data);
	return mat;
}

// expected input: mxn matrix binary file named "matrix_m_n", m vector binary file named "vector_m"
int main(int argc, char *argv[])
{	
	if(argc < 3)
		return 0;
	char* matrix_file_name = argv[1];
	char* vector_file_name = argv[2];
	
    int n, m, vec_dim;
    Eigen::SparseMatrix<double> A = read_sparse_matrix(matrix_file_name, n, m);
    Eigen::VectorXd b = read_vector(vector_file_name, vec_dim);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n,1);
	if(vec_dim != m) {
		printf("Vector dimension (%d) must agree with number of rows (%d) in matrix",vec_dim,m);
		return 0;
	}
	/*
	Eigen::MatrixXd A = Eigen::MatrixXd::Random(m,n);
	Eigen::VectorXd b = Eigen::VectorXd::Random(m,1);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n,1);
	*/


    printf("Starting Calculation (n = %d,m = %d)\n",n,m);
    printf("initial residual = %f\n",norm(b));

    auto start = std::chrono::high_resolution_clock::now();
    lsqr(A,b,x);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "x =\n" << x << std::endl;
    printf("elapsed time [s]: %f\n",elapsed.count());
    printf("final residual = %f\n",norm(dot(A,x) - b));
//std::cout << x << std::endl;
    return 0;
}
