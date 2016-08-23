function [newW1,newW2,error,nextStepSize,row_k_f,shrunken,g0s,g1s,g2s,Gs,n,indicesSAG] = method2Step(W1,W2,stepSize,smaller,larger,lb,ub,regularization,images,labels,m,learningRate,idx,g0s,g1s,g2s,Gs,indicesSAG,n)
% first we pick which weights we will focus on
[~,k0] = size(W1);
[k2,k1] = size(W2);
num = k1*k0 + k1*k2;
shrunken = false;


% initialization of sizes
sumError = 0;

gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
g0 = zeros(k0,1);
g1 = zeros(k1,1);
g2 = zeros(k2,1);
h1 = zeros(k1,1);

stop = false;


% for now we will leave the backprop to still calculate the gradient since
% this does not take up too much time

image = images(:,idx);
label = labels(idx);

[error,gradW1,gradW2,g0,g1,g2,~] = forwBackPropKRYLOV(image,W1,W2,label);
    

subsetSize = num;

g = getG(gradW1,gradW2);

%need g0s,g1s,g2s,Gs,indicesSAG,n all passed through 
[g0s,g1s,g2s,Gs,indicesSAG,n] = Add(g0s,g1s,g2s,Gs,indicesSAG,g0,g1,g2,g,idx,n);
if (n > m)
    disp('we gotta problem');
end

[g0,g1,g2,g] = getTotals(g0s,g1s,g2s,Gs,indicesSAG);
h1 = g1;
for i = 1:k1
    gradW1(i,:) = g((i-1)*k0 + 1:i*k0);
end
count = k0*k1;
for i = 1:k2
    gradW2(i,:) = g(count + (i-1)*k1 + 1:count + i*k1);
end
uncompressedg = g;

indices = ones(num,1);

g = compress(g,indices,subsetSize);


for i = 1:k1
    W((i-1)*k0 + 1:i*k0) = W1(i,:);    
end
count = k0*k1;

for i = 1:k2
    W(count + (i-1)*k1 + 1:count + i*k1) = W2(i,:);
end
uncompressedtotalg = uncompressedg + regularization*W.';
totalg = compress(uncompressedtotalg,indices,subsetSize);
W_GD = -learningRate*totalg;

tol = 10^-6;
maxit = 20;
indices = ones(num,1);
[p0,flag] = pcg(@(v)v2Gv(v,g0,g1,g2,h1,W1,W2,indices,regularization),-totalg,tol,maxit);




W1_GD = -learningRate*(gradW1 + regularization*W1);
W2_GD = -learningRate*(gradW2 + regularization*W2);
error = getTotalError(W1,W2,images,labels,m,regularization);





% result is for debugging purposes (result should equal -g)


disp('aboud to do eigs')
W1_p0 = zeros(k1,k0);
W2_p0 = zeros(k2,k1);

% we want to limit # iterations
try    
    numIterations = 20;
    opts.maxit = numIterations;
    [Vs,lambdas] = eigs(@(v)A2Av(v,totalg,stepSize,g0,g1,g2,h1,W1,W2,indices,regularization),2*subsetSize,1,'SR',opts);
catch 
    for i = 1:k1
        W1_p0(i,:) = p0((i-1)*k0 + 1:i*k0);
    end
    count = k1*k0;
    for i = 1:k2 
        W2_p0(i,:) = p0(count + (i-1)*k1 + 1:count + i*k1);
    end
    error0 = getTotalError(W1 + W1_GD,W2+W2_GD,images,labels,m,regularization);    
    errorG = getTotalError(W1 + W1_GD,W2+W2_GD,images,labels,m,regularization);
    if (norm(p0) <= stepSize && error0 <= errorG)
        newW1 = W1 + W1_p0;
        newW2 = W2 + W2_p0;
    else
        
        newW1 = W1 + W1_GD;
        newW2 = W2 + W2_GD;
    end
    nextStepSize = stepSize;
    totalError = getTotalError(W1,W2,images,labels,m,regularization);
    row_k_f = 0;
    stepped = true;
    GDStep = true;
    disp('error');
    disp(totalError);
    return;
end

disp('finished eigs');
Vs = real(Vs);

lambdas = real(diag(lambdas));

mindex = indexAtMin(lambdas);
lambda = lambdas(mindex);
if (lambda*stepSize^2 > -10^-10)
    disp('lambda ~0, approximate min, step size:');
    disp(stepSize);
    stop = true;
end
V = real(Vs(:,mindex));
[nn,~] = size(V);


y1 = V(1:subsetSize);

y2 = V(subsetSize+1:nn);
hardcase = false;
if (norm(y1) < 10^-4)

    hardcase = true;
    disp('HARD CASE');
    %stop = true;
    %return;
end

%options for p1;
% this should be negative

p1 = -sign(g.'*y2)*stepSize*y1/norm(y1);
compressed_p1 = p1;
p1 = decompress(p1,indices);


if isnan(p0(1))
    p0 = 0*p0;
end


p = zeros(size(p1));


% model_s should aLWAYS be negative I believe




% this is questionable and maybe should be removed (it is essentially a
% hack
W1_p1 = zeros(k1,k0);
W2_p1 = zeros(k2,k1);


for i = 1:k1
    W1_p1(i,:) = p1((i-1)*k0 + 1:i*k0);
    W1_p0(i,:) = p0((i-1)*k0 + 1:i*k0);
end
count = k1*k0;
for i = 1:k2
    W2_p1(i,:) = p1(count + (i-1)*k1 + 1:count + i*k1); 
    W2_p0(i,:) = p0(count + (i-1)*k1 + 1:count + i*k1);
end

error = getTotalError(W1,W2,images,labels,m,regularization);
error1 = getTotalError(W1+W1_p1,W2+W2_p1,images,labels,m,regularization);
error0 = getTotalError(W1+W1_p0,W2+W2_p0,images,labels,m,regularization);
errorGD = getTotalError(W1+W1_GD,W2+W2_GD,images,labels,m,regularization);

if isnan(error0)
    disp('stophere');
end
% model_s should aLWAYS be negative I believe

if ((errorGD < error0 || norm(p0) > stepSize)&& errorGD < error1)
    p = W_GD;
    newerror = errorGD;
    newW1 = W1_GD + W1;
    newW2 = W2_GD + W2;
else if (error1 < error0 || isnan(error0))
        p = p1;
        newerror = error1;
        newW1 = W1_p1 + W1;
        newW2 = W2_p1 + W2;
    else
        p = p0;
        newerror = error0;
        newW1 = W1_p0 + W1;
        newW2 = W2_p0 + W2;
    end
end


model_s = totalg.'*p + 0.5*p.'*v2Gv(p,g0,g1,g2,h1,W1,W2,indices,regularization);
row_k_f = (newerror - error)/model_s;

% this is questionable and maybe should be removed (it is essentially a
% hack

multiplierdown = (1 - (1-smaller)*(n/m));

multiplierup = (1 + (larger - 1)*(n/m));
if (row_k_f <= lb)   
    nextStepSize = smaller*stepSize;
    disp('smaller by:');
    disp(multiplierdown);
    shrunken = true;
    newW1 = W1;
    newW2 = W2;

else    
    if (row_k_f < ub || stepSize > 2)
        nextStepSize = stepSize;
    else
        nextStepSize = larger*stepSize;
    end
end
error = getTotalError(W1,W2,images,labels,m,regularization);
disp(error);
disp(error1);
disp(error0);
disp(errorGD);
disp(newerror);
if (stepSize > 10)
    disp('stopped here');
end

if (isnan(newerror))
    newW1 = W1;
    newW2 = W2;
end

if (newerror > error)
    disp('BAD');
end

end

function [g0s,g1s,g2s,Gs,indices,n] = Add(g0s,g1s,g2s,Gs,indices,g0,g1,g2,G,index,n)
    if (indices(index) == 0)
        indices(index) = 1;
        n = n + 1;
    end
    g0s(:,index) = g0;
    g1s(:,index) = g1;
    g2s(:,index) = g2;
    Gs(:,index) = G;
end

function [g0,g1,g2,G] = getTotals(g0s,g1s,g2s,Gs,indices)
    [g0rows,cols] = size(g0s);
    [g1rows,~] = size(g1s);
    [g2rows,~] = size(g2s);
    [Grows,~] = size(Gs);
    g0 = zeros(g0rows,1);
    g1 = zeros(g1rows,1);
    g2 = zeros(g2rows,1);
    G = zeros(Grows,1);
    
    count = 0;
    for i = 1:cols
        if (indices(i) == 1)
            count = count + 1;
            g0 = g0 + g0s(:,i);
            g1 = g1 + g1s(:,i);
            g2 = g2 + g2s(:,i);
            G = G + Gs(:,i);
        end
    end
    g0 = g0 / count;
    g1 = g1 / count;
    g2 = g2 / count;
    G = G / count;
end