function somTrain(patterns)

global  maxNeighborDist tuneND orderLR orderSteps tuneLR      ;

cases=size(patterns,2);

%ORDERING
neighborDist=maxNeighborDist;
learningRate=orderLR;
for ii=1:orderSteps

for i=1:cases
    pattern=patterns(:,i);
    somUpdate(pattern,learningRate,neighborDist);
end
neighborDist=max(fix(neighborDist/2),tuneND);
learningRate=max(learningRate/2,tuneLR);

end

%TUNING

tuningCoeff=5;
tuningSteps=tuningCoeff*orderSteps; %set number of epochs
learningRate=tuneLR;
neighborDist=tuneND;

for ii=1:tuningSteps

for i=1:cases
    pattern=patterns(:,i);
    somUpdate(pattern,learningRate,neighborDist);
end

learningRate=learningRate*0.99; %adjust lr, comment to keep it constant

end
