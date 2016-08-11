function [newimages,newlabels] = randomSet(subset,totalSet,images,labels)

[~,array] = randiVector(subset,totalSet);

[n,~] = size(images);
%m is number of columns
newimages = zeros(n,subset);
newlabels = zeros(subset,1);
index = 1;
for i = 1:totalSet
    
    if (array(i) == 1)
        newimages(:,index) = images(:,i);
        newlabels(index) = labels(i);
        index = index + 1;
    end
    
end