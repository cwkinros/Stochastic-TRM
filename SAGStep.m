function [newW1,newW2,error,Gs,n,indicesSAG,g] = SAGStep(W1,W2,regularization,images,labels,m,learningRate,idx,Gs,indicesSAG,n)

% first we pick which weights we will focus on
[~,k0] = size(W1);
[k2,k1] = size(W2);

% initialization of sizes

% for now we will leave the backprop to still calculate the gradient since
% this does not take up too much time

image = images(:,idx);
label = labels(idx);

[~,gradW1,gradW2] = forwBackProp(image,W1,W2,label);
    


g = getG(gradW1 + regularization*W1,gradW2 + regularization*W2);

%need g0s,g1s,g2s,Gs,indicesSAG,n all passed through 
[Gs,indicesSAG,n] = Add(Gs,indicesSAG,g,idx,n);

g = getTotals(Gs,indicesSAG);

for i = 1:k1
    gradW1(i,:) = g((i-1)*k0 + 1:i*k0);
end
count = k0*k1;
for i = 1:k2
    gradW2(i,:) = g(count + (i-1)*k1 + 1:count + i*k1);
end


W1_GD = -learningRate*gradW1;
W2_GD = -learningRate*gradW2;

error = getTotalError(W1,W2,images,labels,m,regularization);

errorGD = getTotalError(W1+W1_GD,W2+W2_GD,images,labels,m,regularization);

newW1 = W1 + W1_GD;
newW2 = W2 + W2_GD;

disp(error);
disp(errorGD);
end

function [Gs,indices,n] = Add(Gs,indices,G,index,n)
    if (indices(index) == 0)
        indices(index) = 1;
        n = n + 1;
    end
    Gs(:,index) = G;
end

function [G] = getTotals(Gs,indices)
    [Grows,cols] = size(Gs);
    G = zeros(Grows,1);
    
    count = 0;
    for i = 1:cols
        if (indices(i) == 1)
            count = count + 1;
            G = G + Gs(:,i);
        end
    end
    G = G / count;
end