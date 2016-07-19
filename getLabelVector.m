function [vector] = getLabelVector(label)

if (label == 0)
    label = 10;
end

identity = eye(10);

vector = identity(:,label);