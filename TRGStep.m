function [newW1,newW2,error,nextStepSize,row_k_f,stop,stepped] = TRGStep(W1,W2,images,labels,stepSize,m,print,smaller,larger,lb,ub,maxStepSize,subsetSize,indices)



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

g = compress(g,indices,subsetSize);

gradStepW1 = -stepSize*gradW1 - regularization*W1;
gradStepW2 = -stepSize*gradW2 - regularization*W2;

for i = 1:k1
    W((i-1)*k0 + 1:i*k0) = W1(i,:);    
end
count = k0*k1;

for i = 1:k2
    W(count + (i-1)*k1 + 1:count + i*k1) = W2(i,:);
end


% result is for debugging purposes (result should equal -g)

p0 = pcg(@(v)v2Gv(v,g0,g1,g2,h1,W1,W2,indices),-g);
result = v2Gv(p0,g0,g1,g2,h1,W1,W2,indices);



disp('aboud to do eigs')

if (norm(g) < 10^-4)
    newW1 = W1;
    newW2 = W2;
    error = 0;
    nextStepSize = stepSize;
    row_k_f = 0;
    return;
end
[Vs,lambdas] = eigs(@(v)A2Av(v,g,stepSize,g0,g1,g2,h1,W1,W2,indices),2*subsetSize,1,'SR');

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
compressed_p0 = p0;
compressed_p1 = p1;
p0 = decompress(p0,indices);
p1 = decompress(p1,indices);
%p1 = p1_attempt2;
%disp(p1);
if print    
    disp('reformating p0 and p1 to W1 and W2 sets');
end

for i = 1:k1
    W1_p1(i,:) = p1((i-1)*k0 + 1:i*k0);
    W1_p0(i,:) = p0((i-1)*k0 + 1:i*k0);
end
count = k1*k0;
for i = 1:k2
    W2_p1(i,:) = p1(count + (i-1)*k1 + 1:count + i*k1); 
    W2_p0(i,:) = p0(count + (i-1)*k1 + 1:count + i*k1);
end

error = getTotalError(W1,W2,images,labels,m);


newerror = 0;
p = zeros(size(p1));
if (norm(p0) >= stepSize)

    disp('previous error:');

    disp('new error:');
    error1 = getTotalError(W1 + W1_p1,W2 + W2_p1,images,labels,m);
	error0 = getTotalError(W1 + W1_p0,W2 + W2_p0,images,labels,m);

    disp('previous error followed by error1 and error0');
    disp(error);
    disp(error1);
    disp(error0);
    
    
    if (true)%error >= error1)
        newW1 = W1 + W1_p1;
        newW2 = W2 + W2_p1;
        p = compressed_p1;
        newerror = error1;
    end
    
    disp('p1');
    
else 
    % must check error vals
    disp('must GET TOTAL ERROR for both options');
    error0 = getTotalError(W1 + W1_p0,W2 + W2_p0,images,labels,m);
    error1 = getTotalError(W1 + W1_p1,W2 + W2_p1,images,labels,m);
    disp('previous error followed by error1 and error0');
    disp(error);
    disp(error1);
    disp(error0);
    if (error1 >= error0 || isnan(error1))% && error >= error0)
        newW1 = W1_p0 + W1;
        newW2 = W2_p0 + W2;
        newerror = error0;
        p = compressed_p0;
        disp('p0');
    else if (true)%error >= error1)
        newW1 = W1_p1 + W1;
        newW2 = W2_p1 + W2;
        newerror = error1;
        p = compressed_p1;
        disp('p1');
        end
    end
    
end
% model_s should aLWAYS be negative I believe

model_s = g.'*p + 0.5*p.'*v2Gv(p,g0,g1,g2,h1,W1,W2,indices);
row_k_f = (newerror - error)/model_s;

% this is questionable and maybe should be removed (it is essentially a
% hack

stepped = false;
if (row_k_f <= lb)   
%    nextStepSize = smaller*stepSize;
    nextStepSize = stepSize;
    newW1 = W1;
    newW2 = W2;    
    newerror = error;
else    
    stepped = true;
    if (row_k_f < ub || stepSize >= maxStepSize)
        nextStepSize = stepSize;
    else
        nextStepSize = larger*stepSize;
    end
end


if (newerror > error)
    disp('BAD');
end
%this means we ALWAYS go to the global minima
%newW1 = W1_p0;
%newW2 = W2_p0;


% we need a funtion that inputted a vector outputs Av -> where A is the
% pencil



% now we have what we need for the v -> Hv function




