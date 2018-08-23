function [a] = somActivation(pattern,neighborDist)

 

out=somOutput(pattern); %get winner
index=find(out==1); %find winer's index
geitones=[1:neighborDist]; %distances
GeitonikaIndices=[index(1)-geitones,index(1)+geitones]; %get their indices
NormalisedIndices=GeitonikaIndices( ( GeitonikaIndices>0 )& GeitonikaIndices < (numel(out)) );
%keep those greater or equal to one
out(NormalisedIndices)=0.5; %and now assign to 'em the proper value
a=out;

