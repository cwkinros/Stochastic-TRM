function [nextW1,nextW2,error] = GradDescentStep(W1,W2,images,labels,stepSize,m,regularization)

sumError = 0;
gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
for i = 1:m
    image = images(:,i);
    label = labels(i);
    
    [errori,gradW1i,gradW2i] = forwBackProp(image,W1,W2,label);
    gradW1 = gradW1 + gradW1i;
    gradW2 = gradW2 + gradW2i;
    sumError = sumError + errori;
end
gradW1 = gradW1 / m;
gradW2 = gradW2 / m;
error = sumError / m;

nextW1 = W1 - stepSize*gradW1 - regularization*W1;

nextW2 = W2 - stepSize*gradW2 - regularization*W2;

