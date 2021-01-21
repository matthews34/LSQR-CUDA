#pragma once

#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "matmul\SpMat.h"
#include "matmul\GPUVector.h"
#include "lsqr_gpu.h"


template<typename Vec>
double norm(Vec &v) {
    double norm = v.norm();
	return norm;
}

template<typename Mat>
Mat transpose(Mat &m) {
    return m.transpose();
}

template<typename Mat, typename Vec>
Vec dot(Mat &A, Vec& b) {
    Vec y = A*b;
	return y;
}

template<typename Vec>
Vec scale(Vec& v, double s) {
    Vec y = v*s;
	return y;
}

template<typename Vec>
int size(Vec& x) {
    return x.size();
}

template<typename Vec>
Vec add(Vec& a, Vec& b) {
	Vec c = a + b;
	return c;
}

template<typename Vec>
Vec sub(Vec& a, Vec& b) {
	Vec c = a - b;
	return c;
}

template<typename Mat, typename Vec>
void lsqr(Mat& A, Vec& b, Vec& x) {
	printf("Starting lsqr()\n");
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
    //int it_max = size(x);
	int it_max = 2;
	double epsilon = 0.001;
    double rho, phi, c, s, theta, residual;
    for(int i = 0; i < it_max; i++) {
        // (3) Bidiagonalization
		Vec scal_u;
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
        if(residual < epsilon) {
            printf("finished after %d iterations\n",i);
            return;
        }
    }
    printf("it_max exeeded\n");
}