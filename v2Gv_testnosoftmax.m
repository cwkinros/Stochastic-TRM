function [Gv] = v2Gv_testnosoftmax(v,g0,g1,g2,h1,W1,W2)



[rows1,cols1] = size(W1);
[rows2,cols2] = size(W2);

W1_1 = zeros(rows1,cols1);
W2_1 = zeros(rows2,cols2);

% matricize
for i = 1:rows1
    W1_1(i,:) = v((i-1)*cols1 + 1:i*cols1);    
end
count = rows1*cols1;

for i = 1:rows2
    W2_1(i,:) = v(count + (i-1)*cols2 + 1:count + i*cols2);
end
[W1_2,W2_2] = v2Gv_average_test(g0,g1,g2,h1,W1_1,W2_1,W1,W2);

% vectorize

for i = 1:rows1
    Gv((i-1)*cols1 + 1:i*cols1) = W1_2(i,:);    
end
count = rows1*cols1;

for i = 1:rows2
    Gv(count + (i-1)*cols2 + 1:count + i*cols2) = W2_2(i,:);
end

Gv = Gv.';