function [error, gradW1, gradW2, g0, g1, g2, h1] = forwBackPropKRYLOV(x,W1,W2,label)


% W1 is size hiddenlayersize x input size (200 x 784)
% W2 is size output size x hiddenlayersize (10 x 200)

%-----------forward prop---------------
g0 = x;
h1 = W1*x;
g1 = h1;
h2 = W2*g1;
g2 = Softmax(h2);
output = g2;
y = getLabelVector(label);
error = output - y;
%---------forward prop ends-------------

%----------back prop------------------------
error_h2 = (error).*(g2.*(ones(size(g2))-g2));
gradW2 = error_h2*g1.';
error_h1 = error_h2.'*W2;
gradW1 = error_h1.'*x.';
error = error.'*error;
%-----------back prop ends----------------