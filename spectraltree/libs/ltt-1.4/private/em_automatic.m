function [CPTs, ll, best_K, opt] = em_automatic(x, nsyms, kids, opt);
% run some EM-procedure 
%
%  all kids will be conditionally independent form all variables (including
%  the kids themselves) but excluding itself.
%
%  inputs:
%    x        is a cell array of all data
%    nsyms    a list of the number of symbols
%    kids     is a list of the children
%
%  outputs:
%    CPTs     is a structure with the learned CPTs
%    ll       is the log likelihood
%    best_K   is the number of hidden states learned
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('opt', 'var'), opt = []; end
if ~isfield(opt, 'tol'),      opt.tol      = 1e-5; end
if ~isfield(opt, 'restarts'), opt.restarts = 10; end
if ~isfield(opt, 'maxiter'),  opt.maxiter  = 100; end
if ~isfield(opt, 'verbose'),  opt.verbose  = 0; end
if ~isfield(opt, 'binarysearch'),  opt.binarysearch  = 1; end
if ~isfield(opt, 'Kmax'),  opt.Kmax  = 1025; end

Kmax = opt.Kmax;

N = opt.N;  % number of data points
bic = -Inf;         % overall the best BIC reached
best_K = NaN;
K = 1;
K_inc = 0.5;
K_factor = 2;
while K <= Kmax
  if opt.verbose > 0
    fprintf('[%s.m] binary search: trying K == %d\n', mfilename, K);
  end
  if K==1
    opt_alt = opt;
    opt_alt.restarts = 1;
    [l_CPTs, l_ll] = em_with_restarts(K, x, nsyms, kids, opt_alt);
  else
    [l_CPTs, l_ll] = em_with_restarts(K, x, nsyms, kids, opt);
  end

  % calculate local BIC score
  l_par = l_CPTs.df;
  l_bic = l_ll - l_par * log2(N) / 2;
  if opt.verbose > 0
    fprintf('[%s.m] K==%d ll==%f BIC==%f df==%d N==%d\n', mfilename, K, l_ll, l_bic, l_par, N);
  end
  
  % check whether the latest restart reached a larger BIC
  if l_bic > bic
    best_K = K;
    bic    = l_bic;
    ll     = l_ll;
    CPTs   = l_CPTs;
    if opt.binarysearch == 1
      % do binary search
      K_inc = K_factor * K_inc;
      if K_inc < 1, break, end
      K = K + K_inc;
    else
      K = K + 1;
    end
  else
    if opt.binarysearch == 0
      % don't do binary search
      break
    end
    % go back
    if K_factor == 2
      K_factor = 0.5;
    end
    K_inc = K_factor * K_inc;
    if K_inc < 1, break, end
    K = K - K_inc;
  end
end
if K > Kmax
  % we couldn't find anything below Kmax, thus take the largest/last  
  if opt.binarysearch == 1 
    best_K = K - K_inc;
  else
    best_K = K - 1;
  end
  CPTs = l_CPTs;
  ll   = l_ll;
end
if opt.verbose > 0
  fprintf('[%s.m] FINAL K==%d ll==%f BIC==%f\n', mfilename, best_K, ll, bic);
end
