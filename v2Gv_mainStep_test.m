function [W1_2hat, W2_2hat] = v2Gv_mainStep_test(g0,g1,g2,h1,W1_1,W2_1,W1,W2)

% first calculate g2_2hat = g2_1

g0_1 = 0*g0;
%size(g0_1)
%size(W1)
%size(W1_1)
%size(g0)

h1_1 = W1*g0_1 + W1_1*g0;
g1_1 = h1_1;

%[h1_n,~] = size(h1);
%disp(h1_n);

%for i = 1:h1_n
%    if (h1(i) == 0)
%        g1_1(i) = 0;
%    end
%end

%disp(size(W2));
%disp(size(g1_1));
%disp(size(W2_1*g1));
h2_1 = W2*g1_1 + W2_1*g1;
g2_1 = h2_1;

g2_2hat = g2_1;

% now calculate the final product

h2_2hat = g2_2hat;
g1_2hat = W2.'*h2_2hat;
W2_2hat = h2_2hat*g1.';

h1_2hat = g1_2hat; %times ones and zeros

%[h1_n,~] = size(h1);

%for i = 1:h1_n
%    if (h1(i) == 0)
%        h1_2hat(i) = 0;
%    end
%end

%this should be an average, not everything else
W1_2hat = h1_2hat*g0.';






