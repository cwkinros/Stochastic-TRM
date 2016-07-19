function [theta] = vectorize_layers2(W1,W2)

[r1,c1] = size(W1);
[r2,c2] = size(W2);

theta = zeros(r1*c1 + r2*c2,1);
count = 1;
for i = 1:r1
    newcount = count + c1 - 1;
    theta(count:newcount) = W1(i,:);
    count = newcount + 1;
end
for i = 1:r2
    newcount = count + c2 - 1;
    theta(count:newcount) = W2(i,:);
    count = newcount + 1;
end