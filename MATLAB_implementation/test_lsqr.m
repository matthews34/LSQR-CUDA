cla
n = 2;
A = randn(n,n)*0.5;
b = randn(n,1);
% A = [1,0;0,0];

[x,history] = lsqr_algo(A,b);
[x_2,history_2] = grad_desc(A,b);

[X,Y] = meshgrid(x(1)-8:0.1:x(1)+8,x(2)-8:0.1:x(2)+8);
Z = zeros(size(X));
for i = 1:length(X)
    for j = 1:length(Y)
        Z(i,j) = norm(A*[X(i,j);Y(i,j)] - b);
    end
end
contourf(X,Y,Z)
hold on
scatter(history(1,1:end-1),history(2,1:end-1),'w')
scatter(history(1,end),history(2,end),'x','w')
scatter(history_2(1,end),history_2(2,end),'x','r')
p1 = line(history_2(1,:),history_2(2,:),'Color','r');
p2 = line(history(1,:),history(2,:),'Color','w');
legend([p1,p2],{'Gradient Descent','LSQR'})

axis tight
xlabel('x_1')
ylabel('x_2')
