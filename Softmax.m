function [fx] = Softmax(x)

sum = 0;

[n,m] = size(x);
if (m > n)
    n = m;
end
fx = zeros(n,1);

for i = 1:n
    fx(i) = exp(x(i));
    sum = sum + fx(i);
end

fx = fx./sum;