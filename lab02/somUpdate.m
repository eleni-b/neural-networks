function somUpdate(pattern,learningRate,neighborDist)

global           IW ;


a= somActivation(pattern,neighborDist);
%vectorised code for best performance
alpha_DIW= ( (pattern)*ones(1,size(IW,1))-IW' ) .* ( a * ones(1,size(IW,2)) )' ; %a(i)*(x-w(i))
IW=IW+learningRate*alpha_DIW';

