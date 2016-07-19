function [v,array] = randiVector(size,max)

% returns a vector of UNIQUE values

array = zeros(max,1);

v = zeros(size,1);

for i = 1:size
    notready = true;
    while (notready)
        integer = randi([1 max],1);
        if (array(integer) == 0)
            notready = false;
        end
    end
    v(i) = integer;
    array(integer) = 1;
end

