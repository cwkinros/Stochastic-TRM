function [v] = compress(vector,array,newlength)

v = zeros(newlength,1);
j = 1;
for i = 1:newlength
    a = true;
    while (a)
        if (array(j) == 1)
            a = false;
        else
            j = j + 1;
        end
    end
    v(i) = vector(j);
    j = j + 1;
end
