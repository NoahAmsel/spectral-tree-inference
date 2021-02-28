function b = bic(d, t, ll)
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

% calculates the BIC score
if ~exist('ll', 'var')
  ll = forrest_ll_fast(d, t);
end
b = ll - 0.5 * t.df * log2(size(d.x,2));
