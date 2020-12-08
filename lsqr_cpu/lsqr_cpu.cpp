#include <Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <chrono>

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
        if(residual < epsilon) {
            printf("finished after %d iterations\n",i);
            return;
        }
    }
    printf("it_max exeeded\n");
}

int main(int argc, char *argv[])
{
    int n = 1000, m = 800;
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(m,n);
    Eigen::VectorXd b = Eigen::VectorXd::Random(m,1);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n,1);

    printf("Starting Calculation (n = %d,m = %d)\n",n,m);
    printf("initial residual = %f\n",norm(b));

    auto start = std::chrono::high_resolution_clock::now();
    lsqr(A,b,x);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    //std::cout << "A =\n" << A << std::endl;
    //std::cout << "b =\n" << b << std::endl;
    //std::cout << "x =\n" << x << std::endl;
    printf("elapsed time [s]: %f\n",elapsed.count());
    printf("final residual = %f\n",norm(dot(A,x) - b));
    return 0;
}
