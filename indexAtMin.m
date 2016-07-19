function [mindex] = indexAtMin(vector)

min = Inf;

[n,m] = size(vector);
if (m > n)
    n = m;
end

mindex = 0;

for i = 1:n
    if (vector(i) < min)
        min = vector(i);
        mindex = i;
    end
end