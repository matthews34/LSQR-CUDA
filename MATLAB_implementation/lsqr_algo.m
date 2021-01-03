function [x,history] = lsqr_algo(A,b)
% implementation of the lsqr algorithm
% (1) initialization
beta = norm(b);
u = b/beta;
v = A'*u;
alpha = norm(v);
v = v/alpha;
w = v;
x = 0;
phi_hat = beta;
rho_hat = alpha;
% (2) iterate
it_max = 10;
epsilon = 10^-3;
history = zeros(length(b),0);
history(:,end+1) = x;
for i = 1:it_max
    % (3) bidiagonalization
    u = A * v - alpha * u;
    beta = norm(u);
    u = u / beta;
    v = A' * u - beta * v;
    alpha = norm(v);
    v = v / alpha;
    % (4) orthogonal transformation
    rho = sqrt(rho_hat^2 + beta^2);
    c = rho_hat / rho;
    s = beta / rho;
    theta = s * alpha;
    rho_hat = -c * alpha;
    phi = c * phi_hat;
    phi_hat = s * phi_hat;
    % (5) update x, w
    x = x + (phi / rho) * w;
    w = v - (theta / rho) * w;
    history(:,end+1) = x;
    residual = norm(A*x - b);
    if(residual < epsilon)
        disp(['terminated after ',num2str(i),' iterations'])
        disp(['final residual: ',num2str(residual)])
        return
    end
end
disp(['it_max (=',num2str(it_max),') reached'])
disp(['final residual: ',num2str(residual)])
end

