function [g] = getG(gradW1,gradW2)

[rows1,cols1] = size(gradW1);
[rows2,cols2] = size(gradW2);

g = zeros(rows1*cols1 + rows2*cols2,1);

for i = 1:rows1
    g((i-1)*cols1 + 1:i*cols1) = gradW1(i,:);
end

count = rows1*cols1;

for i = 1:rows2
    g(count + (i-1)*cols2 + 1:count + i*cols2) = gradW2(i,:);
end
