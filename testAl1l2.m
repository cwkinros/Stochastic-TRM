function [] = testAl1l2()

disp('test1:');
W1 = 2;
W2 = 10;
x = 1;
J = getJacobi(W1,W2,x);
G = J.'*J;

W1_1 = 10/sqrt(2);
W2_1 = 2/sqrt(2);

Theta1 = vectorize_layers2(W1_1,W2_1);
Theta2_gt = G*Theta1;

[GW1,GW2] = Algorithm1_layers2(W1,W2,W1_1,W2_1,x);
Theta2_a1 = vectorize_layers2(GW1,GW2);

disp('ground truth beside a1 result:');
disp([Theta2_gt(:,1),Theta2_a1(:,1)]);
passed = true;
for i = 1:2
    if (Theta2_gt ~= Theta2_a1)
        passed = false;
    end
end
if (passed)
    disp('test 1 passed');
else 
    disp('test 1 failed');
end



disp('test 2:');

W1 = [1,2,3;4,5,6];
W2 = [3,4;2,4;1,2];
x = [1;2;3];
J = getJacobi(W1,W2,x);
G = J.'*J;


W1_1 = rand(2,3);
W2_1 = rand(3,2);
W1_1 = W1_1*(1/(sum(sum(W1_1)) + sum(sum(W2_1))));
W2_1 = W2_1*(1/(sum(sum(W1_1)) + sum(sum(W2_1))));
disp(W1_1);



Theta1 = vectorize_layers2(W1_1,W2_1);
Theta2_gt = G*Theta1;

[GW1,GW2] = Algorithm1_layers2(W1,W2,W1_1,W2_1,x);
Theta2_a1 = vectorize_layers2(GW1,GW2);

disp('ground truth beside a1 result:');
disp([Theta2_gt(:,1),Theta2_a1(:,1)]);
passed = true;
for i = 1:12
    if (Theta2_gt ~= Theta2_a1)
        passed = false;
    end
end
if (passed)
    disp('test 2 passed');
else 
    disp('test 2 failed');
end




