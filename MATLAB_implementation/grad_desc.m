function [x,history] = grad_desc(A,b)
%GRAD_DESC Summary of this function goes here
%   Detailed explanation goes here
it_max = 10;
x = zeros(2,1);
epsilon = 10^-3;
history = zeros(length(b),0);
history(:,end+1) = x;
for i =1:it_max
    d = grad(A,x,b);
    history(:,end+1) = x;
    residual = norm(A*x-b);
    fun = @(a)norm(A*(x-a*d) - b);
    a = fminsearch(fun,1/i);
    x = x - a*d;
    disp(['residual ',num2str(residual)])
    if(residual<epsilon)
        disp(['terminated calculation with residual ',num2str(residual)])
        return
    end
end
disp('it_max oversteped')
end

function val = grad(A,x,b)
val = 2*A'*(A*x-b);
end