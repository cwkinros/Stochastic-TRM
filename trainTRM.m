function [W1_M3,W2_M3] = trainTRM(W1,W2)

% images: 784 by 60000 matrix of pixels
images = loadMNISTImages('train-images.idx3-ubyte');

% labels: 60000 by 1 matrix (vector) of labels 
labels = loadMNISTLabels('train-labels.idx1-ubyte');

% number of inputs k0
%k0 = 784;
start = 300;

k0 = 50;

images = images(start+1:start+k0,:);

% number of hidden nodes k1
k1 = 10;

% number of outputs k2 
k2 = 10;

% number of examples 
m = 1000;


W1_TRG = W1;
W2_TRG = W2;
W1_GD = W1_TRG;
W2_GD = W2_TRG;
W1_M2 = W1;
W2_M2 = W2;
W1_M3 = W1;
W2_M3 = W2;

totalWeights = k0*k1 + k1*k2;
%subsetSize = 50;
%subsetSize = 100;
trainingSubsetSize = 400;

a = 90;
b = 1000;
numberIterations_TRG = 1000;%00;%500;
numberIterations_GD = 19*numberIterations_TRG;
numberIterations_M3 = 800;%numberIterations_TRG;

regularization = 0.0000001;
errors_TRG = zeros(numberIterations_TRG,1);
errors_GD = zeros(numberIterations_GD,1);
errors_M2 = zeros(numberIterations_M3,1);
weights = zeros(numberIterations_GD,1);
ballSizes = zeros(numberIterations_TRG,1);
row_k_f = zeros(numberIterations_TRG,1);
stepSize = 0.1;
time_sum_TRG = 0;
time_sum_GD = 0;
time_sum_M3 = 0;

smaller = 0.5;
larger = 2;
upperbound = 0.9;
lowerbound = 0.1;
stop = false;
stepped = true;
timesM3 = zeros(numberIterations_M3,1);
timesSTRM = zeros(numberIterations_TRG,1);
timesGD = zeros(numberIterations_GD,1);
GDCount = 0;
previous_rho = 0;



errors_M3 = zeros(numberIterations_M3,1);


timesM3 = zeros(numberIterations_M3,1);


numrhos = 4;


stepSizes = zeros(numberIterations_M3,1);


time_sum_M3 = 0;
W1_M2 = W1;
W2_M2 = W2;
previous_rho = 0;
trainingSubsetSize = 60;
rows = zeros(numrhos,1);
index = 1;
count = 0;
sum_rho = 0;

%got to save an array of all the things... mm

g0s = zeros(k0,m);
g1s = zeros(k1,m);
g2s = zeros(k2,m);
Gs = zeros(totalWeights,m);
num = 0;
indicesSAG = zeros(m,1);
disp('hello begin');
size_W_M3 = zeros(numberIterations_M3,1);
for i = 1:numberIterations_M3
    tic;
    randIdx = randi(m);
    if (randIdx == 0)
        disp('idx is 0');
        break;
    end
    learningRate = a / (i + b);
    [W1_M3,W2_M3,errors_M3(i),nextStepSize,row_k_f,shrunken,g0s,g1s,g2s,Gs,num,indicesSAG] = method2Step(W1_M3,W2_M3,stepSize,smaller,larger,lowerbound,upperbound,regularization,images,labels,m,learningRate,randIdx,g0s,g1s,g2s,Gs,indicesSAG,num);
    previous_rho = row_k_f;
    stepSizes(i) = stepSize;
    stepSize = nextStepSize;
    time_sum_M3 = time_sum_M3 + toc;
    size_W_M3(i) = sum(sum(abs(W1_M3))) + sum(sum(abs(W2_M3)));
    disp('n:');
    disp(num);
    timesM3(i) = time_sum_M3;
    disp('iterate:');
    disp(i);
end
figure
plot(1:numberIterations_M3,errors_M3);
title('error');

figure
plot(1:numberIterations_M3,stepSizes);
title('stepsizes');
disp('hello done');

trainingSubsetSize =400;
return;

for i = 1:numberIterations_TRG;

    tic;
 
    %[~,indices] = randiVector(subsetSize,totalWeights);
    subsetSize = totalWeights;
    indices = ones(totalWeights,1);
    learningRate = a / (i + b);
    [setImages,setLabels] = randomSet(trainingSubsetSize,m,images,labels);
    [W1_TRG,W2_TRG,errors_TRG(i),stepSize,row_k_f(i),stop,GDStep] = TRGStep(W1_TRG,W2_TRG,setImages,setLabels,stepSize,trainingSubsetSize,false,smaller,larger,lowerbound,upperbound,maxStepSize,subsetSize,indices,regularization,learningRate,images,labels,m);
    disp('iteration:');
    disp(i);
    if (GDStep)
        GDCount = GDCount + 1;
    end
    time_sum_TRG = time_sum_TRG + toc;
    timesSTRM(i) = time_sum_TRG;
    ballSizes(i) = stepSize;

end


trainingSubsetSize = 50;
b = b*3;
for i = 1:numberIterations_GD;
    tic;
    [setImages,setLabels] = randomSet(trainingSubsetSize,m,images,labels);

    learningRate = a / (i + b);
    %learningRate = 1;
    [W1_GD,W2_GD,errors_GD(i)] = GradDescentStep(W1_GD,W2_GD,setImages,setLabels,learningRate,trainingSubsetSize,regularization,images,labels,m);
    weights(i) = sum(sum(abs(W1_GD))) + sum(sum(abs(W2_GD)));

    if (weights(i) == 0)
        disp('ending prematurely due to weights == 0');
        break;
    end

    if (isnan(W1_GD(1,1)))
        disp('breaking here');
        break;
    end
    
    
    if (mod(i,10) == 0)
        disp('error:');
        disp(errors_GD(i));
        disp('lr');
        disp(learningRate);
        disp('iteration:');
        disp(i);
    end
    time_sum_GD = time_sum_GD + toc;
    timesGD(i) = time_sum_GD;
    %learningRate = learningRate*0.96;
    %if (i > 1 && errors_GD(i-1) < errors_GD(i))
    %    learningRate = learningRate*0.5;
    %else
    %    learningRate = learningRate*1.05;
    %end


end






TestGDvsTRM(W1_GD,W2_GD,W1_TRG,W2_TRG,labels,images,m);

figure
plot(1:numberIterations_GD,errors_GD,1:numberIterations_TRG,errors_TRG,1:numberIterations_M2,errors_M2_400);
legend('GD','M1','M2');
title('SGD plotted with SGD&TRM');
xlabel('# iterations');
ylabel('Objective Value');

figure
plot(timesGD,errors_GD,timesSTRM,errors_TRG,timesM2_400,errors_M2_400);
legend('GD','M1','M2');
title('SGD plotted with SGD&TRM');
xlabel('time (s)');
ylabel('Objective Value');

iter_TRG_t = time_sum_TRG / numberIterations_TRG;
iter_GD_t = time_sum_GD / numberIterations_GD;

disp('time per iteration for trust region: ');
disp(iter_TRG_t);
disp('time per iteration for Gradient descent:');
disp(iter_GD_t);

figure

plot(1:numberIterations_TRG,ballSizes);
title('Changing Ball Size');
xlabel('# iterations');
ylabel('ball size');

%figure

%plot(1:numberIterations_TRG,row_k_f);
%title('row_k_f');
%xlabel('iteration #');


%finally we want to check error from non-test set

disp('CHECK TEST ERRORS');
disp(size(images));
disp(size(labels));
errorTRM = getTotalError(W1_TRG,W2_TRG,images(:,m+1:m+1000),labels(m+1:m+1000),1000,0);
errorGD = getTotalError(W1_GD,W2_GD,images(:,m+1:m+1000),labels(m+1:m+1000),1000,0);
disp('TEST ERROR TRM:');
disp(errorTRM);
disp('TEST ERROR GD:');
disp(errorGD);

disp('Percentage of GD steps taken');
disp(GDCount / numberIterations_TRG);

