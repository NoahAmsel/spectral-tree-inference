function a = aic(d, t, ll)
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

% calculates the AIC score
if ~exist('ll', 'var')
  ll = forrest_ll(d, t);
end
a = ll - t.df;