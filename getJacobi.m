function [J] = getJacobi(W1,W2,x)

[b,a] = size(W1);
[c,b] = size(W2);


[d,~] = size(x);

if (d ~= a)
    disp('we have made an error!');
end

% c should be size of f

% first let's get the Jacobi matrices as 2 3D matrices the size of W1 x c
% and W2 x c

J1 = zeros(b,a);
J2 = zeros(c,b);

J = zeros(c,a*b + b*c);

for k = 1:c
    for j = 1:b
        for i = 1:a
            J1(j,i) = W2(k,j)*x(i,1);
        end
        
        for i = 1:c
            if (k == i)
                J2(i,j) = W1(j,:)*x;
            else
                J2(i,j) = 0;
            end
        end
    end
    vector = vectorize_layers2(J1,J2);
    J(k,:) = vector;
    
end

disp(J);
    