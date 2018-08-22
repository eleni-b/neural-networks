
function perf = CrossEntropy(e, x, pp)

if nargin < 1, error('missing arguments'), end

if ischar(e)
  switch e
    case 'version', perf = 3.0;
    case 'deriv', perf = 'CrossEntropyDeriv';
    case 'name', perf = 'CrossEntropy';
    case 'pnames', perf = { 'targets' };
    case 'defaultParam', perf = struct( 'targets', []);
    otherwise, error('unknown argument')
  end
  return
end

if isa(e,'cell'), e = cell2mat(e); end

if isa(e,'double')
  t = pp.targets; %targets
  y = t - e; %estimations;
  t(t==0) = 1e-7; %safeguard from zeros!?
  y(y==0) = 1e-7;
  perf = sum( sum( t.*log(t./y) ) ); %Cross Entropy definition
else
  error('performance function argument not double')
end