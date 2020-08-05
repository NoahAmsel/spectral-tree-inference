function [CPTs, best_ll] = em_with_restarts(K, x, nsyms, kids, opt);
% estimate the Latent Class Model 
%
%  inputs:
%    x        is a cell array of all data
%    nsyms    a list of the number of symbols
%    kids     is a list of the children
%    K        is number of hidden states
%
%  outputs:
%    CPTs     is a structure with the learned CPTs
%    ll       is the log likelihood
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

verbose  = opt.verbose;
restarts = opt.restarts;

best_ll = -Inf;   % the best log-likelihood reached so far
for round = 1:restarts
  if verbose > 2
    fprintf('[%s.m] ROUND %d/%d  best_ll==%f\n', mfilename, round, restarts, best_ll);
  end
  opt.best_ll = best_ll;
  [l_CPTs, l_ll] = lcm_lambda(K, x, nsyms, kids, opt);

  % check whether the latest restart reached a larger Q-function
  if l_ll > best_ll
    best_ll = l_ll;
    ll    = l_ll;
    CPTs  = l_CPTs;
  end
end
if best_ll == -Inf
  warning('[em_with_restarts.m] could not fit model');
end
