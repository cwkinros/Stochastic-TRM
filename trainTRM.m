function [W1_TRG,W2_TRG] = trainTRM(W1,W2)


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
m =300;

W1_TRG = W1;%ones(k1,k0);
W2_TRG = W2;%ones(k2,k1);
W1_GD = W1_TRG;
W2_GD = W2_TRG;

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
subsetSize = 100;

% randomly pick 10 weights at a time and find minimum 

% network is two FC layers with ReLU in the middle and Softmax at the end
learningRate = 0.14;
numberIterations_GD =0;
numberIterations_TRG = 300;
regularization = 0.00000001;
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
for i = 1:numberIterations_TRG;

    tic;
    if stepped
        [~,indices] = randiVector(subsetSize,totalWeights);
    end
    [W1_TRG,W2_TRG,errors_TRG(i),stepSize,row_k_f(i),stop,stepped] = TRGStep(W1_TRG,W2_TRG,images,labels,stepSize,m,false,smaller,larger,lowerbound,upperbound,maxStepSize,subsetSize,indices,regularization);
    if (stop)
        stepped = true;
        stepSize = maxStepSize;
        %disp('Reached local minimum');
        %break;
    end
    stepped = true;
    time_sum_TRG = time_sum_TRG + toc;

    ballSizes(i) = stepSize;

end

for i = 1:numberIterations_GD;
    tic;
    [W1_GD,W2_GD,errors_GD(i)] = GradDescentStep(W1_GD,W2_GD,images,labels,learningRate,m,regularization);
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
    end
    time_sum_GD = time_sum_GD + toc;
    if (i > 1 && errors_GD(i-1) < errors_GD(i))
        learningRate = learningRate*0.8;
    else
%        learningRate = learningRate*1.1;
    end


end



TestGDvsTRM(W1_GD,W2_GD,W1_TRG,W2_TRG,labels,images,m);

figure
plot(1:numberIterations_GD,errors_GD);
title('Gradient Descent Progress');
xlabel('# iterations');
ylabel('Objective Value');



disp('label:');
disp(labels(1));

figure
plot(1:numberIterations_TRG,errors_TRG);
title('Trust Region Method Progress');
xlabel('# iterations');
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



    


