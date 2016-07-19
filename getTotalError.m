function [error] = getTotalError(W1,W2,images,labels,m)

sum = 0;

for i = 1:m
    errori = getError(W1,W2,images(:,i),labels(i));
    sum = sum + errori;
end

error = sum / m;