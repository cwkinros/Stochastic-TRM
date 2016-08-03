function [error] = getTotalError(W1,W2,images,labels,m,regularization)

sum1 = 0;

for i = 1:m
    errori = getError(W1,W2,images(:,i),labels(i));
    sum1 = sum1 + errori;
end
error = sum1 / m;

error = error + regularization*(sum(sum(W1.*W1)) + sum(sum(W2.*W2)));

