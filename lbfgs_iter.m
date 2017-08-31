function [r,convergeflag] = lbfgs_iter(invH0, g, x)

k = size(x,2);
s = x(:,1:end-1) - x(:,2:end);
t = g(:,1:end-1) - g(:,2:end);
rho = dot(s, t);


if isa(g, 'gpuArray')
    alpha = gpuArray.zeros(k-1, 1);
else
    alpha = zeros(k-1, 1);
end


q = -g(:,1);
for i=1:k-1
    if rho(i)==0; break; end        % converged? or LBFGS not quite work, switch to gradient descent by break here
    alpha(i) = dot(s(:,i),q)/rho(i);
    q = q - alpha(i)*t(:,i);
end

r = invH0*q;

for i=k-1:-1:1
    if rho(i)==0; break; end
    beta = dot(t(:,i), r)/rho(i);
    r = r + s(:,i)*(alpha(i)-beta);
end

