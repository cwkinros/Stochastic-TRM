function [output] = forwardProp(x,W1,W2)


h1 = W1*x;
g1 = h1;

h2 = W2*g1;

%---------------------------softmax changes here-------------------------
g2 = Softmax(h2);

output = g2;