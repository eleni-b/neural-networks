function d = CrossEntropyDeriv(code, e, x, perf, pp)

e_was_double = isa(e,'double');
if e_was_double, e = {e}; end

switch code
  case 'e'
    [rows,cols] = size(e);
    d = cell(rows,cols);
    for i=1:rows
      for j=1:cols
        if ~isempty( e{i,j} )
          t = pp.targets; %pattern tartets
          y = t - e{i,j}; %estimations
          d{i,j} = t ./ y; %Gradient dE/dy= -t/y, where E=CrossEntropy
        end
      end
   end
   if e_was_double, d = d{1}; end

  case 'x'
    d = zeros( size(x) );

  otherwise
    error('unknown argument')
end