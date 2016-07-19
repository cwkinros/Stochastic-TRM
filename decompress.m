function [v] = decompress(vector,array)

[n,m] = size(array);

if (m > n)
    n = m;
end

v = zeros(n,1);
index = 1;

for i = 1:n
    if (array(i) == 0)
        v(i) = 0;
    else
        v(i) = vector(index);
        index = index + 1;
    end
end
   