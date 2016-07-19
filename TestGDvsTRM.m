function [] = TestGDvsTRM(W1_GD,W2_GD,W1_TRM,W2_TRM,labels,images,m)


disp('test GD error');
error_GD = getTotalError(W1_GD,W2_GD,images,labels,m);
disp('test TRM error');
error_TRM = getTotalError(W1_TRM,W2_TRM,images,labels,m);

if (error_GD < error_TRM)
    disp('gradient descent did better');
    disp('error_gd:'); 
    disp(error_GD);
    disp('error_trm:'); 
    disp(error_TRM);
else if (error_TRM < error_GD)
    disp('trust region method did better');
    disp('error_trm:');
    disp(error_TRM);
    disp('error_gd:'); 
    disp(error_GD);
else
    disp('twas a tie');
    disp('error:');
    disp(error_GD);
    end
end
    