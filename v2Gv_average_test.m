function [W1_2hat,W2_2hat] = v2Gv_average_test(g0,g1,g2,h1,W1_1,W2_1,W1,W2)

[n,m] = size(g0);


W1_2hat = 0;
W2_2hat = 0;

for i = 1:m
    [W1_2hati,W2_2hati] = v2Gv_mainStep_test(g0(:,i),g1(:,i),g2(:,i),h1(:,i),W1_1,W2_1,W1,W2);
    W1_2hat = W1_2hat + W1_2hati;
    W2_2hat = W2_2hat + W2_2hati;
end

W1_2hat = W1_2hat / m;
W2_2hat = W2_2hat / m;