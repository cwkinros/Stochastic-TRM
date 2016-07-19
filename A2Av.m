function [Av] = A2Av(v,g,ballSize,g0,g1,g2,h1,W1,W2,indices)

% A is [G,-ggT/ballSize^2;-I, G] 

[n,m] = size(g);
if (m > n)
    n = m;
end

y1 = v(1:n);
y2 = v(n+1:2*n);

Gy2 = v2Gv(y2,g0,g1,g2,h1,W1,W2,indices);
Gy1 = v2Gv(y1,g0,g1,g2,h1,W1,W2,indices);

C = g*g.'/(ballSize*ballSize);

Av(1:n) = Gy1 + -C*y2;
Av(n+1:2*n) = -y1 + Gy2;






