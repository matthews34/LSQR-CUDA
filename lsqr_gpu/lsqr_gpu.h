#pragma once

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "matmul/SpMat.h"
#include "matmul/GPUVector.h"

// template method for getting norm of a vector
template<typename Vec>
double norm(Vec &v) {
    double norm = v.norm();
	return norm;
}

// template method for getting transposed matrix
template<typename Mat>
Mat transpose(Mat &m) {
    return m.transpose();
}

// template method for getting norm of a vector
template<typename Mat, typename Vec>
Vec dot(Mat &A, Vec& b) {
    Vec y = A*b;
	return y;
}

// template method for scaling a vector with a scalar
template<typename Vec>
Vec scale(Vec& v, double s) {
    Vec y = v*s;
	return y;
}

// template method for getting the size of a vector
template<typename Vec>
int size(Vec& x) {
    return x.size();
}

// template method for adding two vectors together
template<typename Vec>
Vec add(Vec& a, Vec& b) {
	Vec c = a + b;
	return c;
}

// template method for subtracting two vectors together
template<typename Vec>
Vec sub(Vec& a, Vec& b) {
	Vec c = a - b;
	return c;
}

//lsqr algorithm using template vectors and matricies
template<typename Mat, typename Vec>
void lsqr(Mat& A, Vec& b, Vec& x) {
	Vec residual_vec;
    residual_vec = dot(A,x) - b;
    double residual = norm(residual_vec);
	printf("Initial residual = %f\n",residual);
    // (1) Initialization
    double beta = norm(b);
    Vec u;
	u = scale(b,1/beta);
    Mat A_t = transpose(A);
    Vec v;
	v = dot(A_t,u);
	double alpha = norm(v);
    v = scale(v,1/alpha);
    Vec w = v;
    double phi_hat = beta;
    double rho_hat = alpha;
    // (2) Iteration
    int it_max = size(x);
	double epsilon = 0.001;
    double rho, phi, c, s, theta;
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
        residual = 0;
        Vec residual_vec = dot(A,x) - b;
        residual = norm(residual_vec);
		// Check if residual small enough
        if(residual < epsilon) {
            printf("finished after %d iterations\n",i);
            return;
        }
    }
    printf("it_max exceeded\n");
}
