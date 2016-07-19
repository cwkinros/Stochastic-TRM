function [error] = getError(W1,W2,input,label)

output = forwardProp(input,W1,W2);
%disp('output');
%disp(output);
label = getLabelVector(label);
%disp('input:');
%disp(input);
%disp('output:');
%disp(output);
error = label - output;

error = error.'*error;