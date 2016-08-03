function [W1_GD,W2_GD] = trainTRM(W1,W2)


%--------------------------------regular mnist version-------------------
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
m =1000;

%W1_TRG = rand(k1,k0);
%W2_TRG = rand(k2,k1);
%W1_GD = W1_TRG;
%W2_GD = W2_TRG;

%--------------------------end of regular mnist-------------------------

%---------------------------simplest example----------------------------

%k0 = 2;
%k1 = 2;
%k2 = 10;
%m = 2;
%images = [1,0;0,1];
%labels = [4,8];
%W1_TRG = rand(k1,k0);
%W2_TRG = rand(k2,k1);
%W1_GD = [1,0;0,1];
%W1_GD = W1_GD / sum(sum(W1_GD));
%W2_GD = [0,0;0,0;0,0;0.4,0;0,0;0,0;0,0;0,1;0,0;0,0]*10;
%W2_GD = W2_GD / sum(sum(W2_GD));
%W1_TRG = W1_GD;
%W2_TRG = W2_GD;

W1_TRG = W1;
W2_TRG = W2;
W1_GD = W1_TRG;
W2_GD = W2_TRG;

totalWeights = k0*k1 + k1*k2;
%subsetSize = 50;
subsetSize = 100;
trainingSubsetSize = 20;

% randomly pick 10 weights at a time and find minimum 

% network is two FC layers with ReLU in the middle and Softmax at the end
%learningRate = 0.04;

% these values got 0.5413 error
%a = 200;
%b = 1500;
% for training subset 20


% these values got 0.6075 error
%a = 30;
%b = 150;
% for training subset 20

% these values got 0.5552 test error and 0.5148 training error
a = 30*3;
b = 150*3;
numberIterations_TRG = 100;
numberIterations_GD =19*numberIterations_TRG;

regularization = 0.0001;
errors_TRG = zeros(numberIterations_TRG,1);
errors_GD = zeros(numberIterations_GD,1);
weights = zeros(numberIterations_GD,1);
ballSizes = zeros(numberIterations_TRG,1);
row_k_f = zeros(numberIterations_TRG,1);
stepSize = 1;
time_sum_TRG = 0;
time_sum_GD = 0;

smaller = 0.5;
larger = 2;
upperbound = 0.75;
lowerbound = 0.25;
maxStepSize = 1;
stop = false;
stepped = true;
timesSTRM = zeros(numberIterations_TRG,1);
timesGD = zeros(numberIterations_GD,1);
GDCount = 0;
for i = 1:numberIterations_TRG;

    tic;
 
    [~,indices] = randiVector(subsetSize,totalWeights);
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

for i = 1:numberIterations_GD;
    tic;
    [setImages,setLabels] = randomSet(trainingSubsetSize,m,images,labels);

    learningRate = a / (i + b);
    [W1_GD,W2_GD,errors_GD(i)] = GradDescentStep(W1_GD,W2_GD,setImages,setLabels,learningRate,trainingSubsetSize,regularization,images,labels,m);
    weights(i) = sum(sum(abs(W1_GD))) + sum(sum(abs(W2_GD)));

    if (weights(i) == 0)
        disp('ending prematurely due to weights == 0');
        break;
    end

    if (isnan(W1_GD(1,1)))
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
plot(1:numberIterations_GD,errors_GD);
title('Gradient Descent Progress');
xlabel('# iterations');
ylabel('Objective Value');

figure
plot(timesGD,errors_GD);
title('Gradient Descent Progress');
xlabel('time');
ylabel('Objective Value');

disp('label:');
disp(labels(1));

figure
plot(1:numberIterations_TRG,errors_TRG);
title('Trust Region Method Progress');
xlabel('# iterations');
ylabel('Objective Value');

figure
plot(timesSTRM,errors_TRG);
title('Trust Region Method Progress');
xlabel('time');
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

figure

plot(1:numberIterations_TRG,row_k_f);
title('row_k_f');
xlabel('iteration #');


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

