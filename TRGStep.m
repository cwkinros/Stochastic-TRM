function [newW1,newW2,totalError,nextStepSize,row_k_f,stop,GDStep] = TRGStep(W1,W2,images,labels,stepSize,m,print,smaller,larger,lb,ub,maxStepSize,subsetSize,indices,regularization,learningRate,totalimages,totallabels,totalm)



% first we pick which weights we will focus on
[~,k0] = size(W1);
[k2,k1] = size(W2);
n = k1*k0 + k1*k2;



% initialization of sizes
sumError = 0;

gradW1 = zeros(size(W1));
gradW2 = zeros(size(W2));
g0 = zeros(k0,m);
g1 = zeros(k1,m);
g2 = zeros(k2,m);
h1 = zeros(k1,m);

stop = false;


% for now we will leave the backprop to still calculate the gradient since
% this does not take up too much time
for i = 1:m
    image = images(:,i);
    label = labels(i);

    [errori,gradW1i,gradW2i,g0i,g1i,g2i,h1i] = forwBackPropKRYLOV(image,W1,W2,label);
    
    g0(:,i) = g0i;
    g1(:,i) = g1i;
    g2(:,i) = g2i;
    h1(:,i) = h1i;
    gradW1 = gradW1 + gradW1i;
    gradW2 = gradW2 + gradW2i;
    sumError = sumError + errori;
end


gradW1 = gradW1 / m;
gradW2 = gradW2 / m;

g = getG(gradW1,gradW2);
uncompressedg = g;


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

stepSize = norm(W_GD);

W1_GD = -learningRate*(gradW1 + regularization*W1);
W2_GD = -learningRate*(gradW2 + regularization*W2);
error = getTotalError(W1,W2,images,labels,m,regularization);

for i = 1:k1
    W((i-1)*k0 + 1:i*k0) = W1(i,:);    
end
count = k0*k1;

for i = 1:k2
    W(count + (i-1)*k1 + 1:count + i*k1) = W2(i,:);
end





% result is for debugging purposes (result should equal -g)


disp('aboud to do eigs')

if (norm(g) < 10^-4)
 errorG = getTotalError(W1 + W1_GD,W2+W2_GD,images,labels,m,regularization);    
    newW1 = W1 + W1_GD;
    newW2 = W2 + W2_GD;
    nextStepSize = stepSize;
    totalError = getTotalError(W1,W2,totalimages,totallabels,totalm,regularization);
    row_k_f = 0;
    stepped = true;
    GDStep = true;
    disp('error');
    disp(totalError);
    
    return;
end


% we want to limit # iterations
try    
    numIterations = 20;
    opts.maxit = numIterations;
    [Vs,lambdas] = eigs(@(v)A2Av(v,totalg,stepSize,g0,g1,g2,h1,W1,W2,indices,regularization),2*subsetSize,1,'SR',opts);
catch 
    
    errorG = getTotalError(W1 + W1_GD,W2+W2_GD,images,labels,m,regularization);    
    newW1 = W1 + W1_GD;
    newW2 = W2 + W2_GD;
    nextStepSize = stepSize;
    totalError = getTotalError(W1,W2,totalimages,totallabels,totalm,regularization);
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
    disp('lambda ~0, approximate min');
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
for i = 1:k1
    W1_p1(i,:) = p1((i-1)*k0 + 1:i*k0);
end
count = k1*k0;
for i = 1:k2
    W2_p1(i,:) = p1(count + (i-1)*k1 + 1:count + i*k1); 
end




newerror = 0;
p = zeros(size(p1));


% model_s should aLWAYS be negative I believe

model_s = 0;%g.'*p1 + 0.5*p1.'*v2Gv(p1,g0,g1,g2,h1,W1,W2,indices);
row_k_f = 0;%(newerror - error)/model_s;

% this is questionable and maybe should be removed (it is essentially a
% hack

error1 = getTotalError(W1 + W1_p1,W2 + W2_p1,totalimages,totallabels,totalm,regularization);
errorGD = getTotalError(W1 + W1_GD,W2 + W2_GD,totalimages,totallabels,totalm,regularization);
%error1 = getTotalError(W1 + W1_p1, W2 + W2_p1, images,labels,m,regularization);
%errorGD = getTotalError(W1 + W1_GD,W2 + W2_GD,images,labels,m,regularization);

if (errorGD < error1)
    nextStepSize = stepSize;
    newW1 = W1 + W1_GD;
    newW2 = W2 + W2_GD;
    GDStep = true;
    disp('GD');
else
    disp('TRM');
    nextStepSize = stepSize;
    newW1 = W1 + W1_p1;
    newW2 = W2 + W2_p1;
    GDStep = false;
end
        
totalError = getTotalError(W1,W2,totalimages,totallabels,totalm,regularization);
disp(totalError);
