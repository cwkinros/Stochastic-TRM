function [GW1, GW2] = Algorithm1_layers2(W1,W2,W1_1,W2_1,x)

%because the weights are different sizes we need to specify the # layers
%let Theta and Theta1 be 3 dimentional


[a,~] = size(x);
v = x;
v1 = x*0;
L = 2;
for l = 1:L
    if (l == 1) 
        W = W1;
        W_1 = W1_1;
    else
        W = W2;
        W_1 = W2_1;
    end
    h = W*v;
    h1 = W*v1 + W_1*v;
    v = h;
    v1 = h1;
end
disp('h_1 and h should be 2sqrt(2) and 4');
disp(h1);
disp(h);

%p = exp(v)*(1/sum(exp(v)));


%v2 = (diag(p(:,1)) - p(:,1)*p(:,1).')*v1;

v2 = v1;
for Lminusl = 1:L
    l = L + 1 - Lminusl;
    h2 = v2;
    if (l == 1)
        W = W1;
        h2 = W.'*h2;
        GW1 = v2*1;
        v2 = h2;
    else
        W = W2;
        h2 = W.'*h2;
        GW2 = v2*2;
        v2 = h2;
    end

end

    
    


