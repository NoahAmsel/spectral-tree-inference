function [CPTs, ll] = lcm_lambda(K, x, nsyms, kids, opt);
% 11. May 2008
% Estimate the Latent Class Model with the correct lambda messages, see
% Pearl's book "Probabilistic Reasoning in Intelligent Systems".
%
%  inputs:
%    K        is number of hidden states
%    x        is a cell array of all data  (the beta messages)
%    nsyms    a list of the number of symbols
%    kids     is a list of the children
%    tol      minimal change for measuring convergence
%    maxiter  maximum number of iterations
%    best_ll  best_ll so far (used for "clever" stopping)
%
%  outputs:
%    CPTs     is a structure with the learned CPTs
%    ll       is the log likelihood
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

tol     = opt.tol;
maxiter = opt.maxiter;
best_ll = opt.best_ll;

N = opt.N;   % number of data points
D = length(kids);    % number of children

% initialize
pie   = normalizzze(0.4+0.2*rand(K, 1));     % p(z) == pie
theta = cell(1, D);                          % p(x|z) == theta
thetabeta = cell(1, D);                      % thetabeta = theta_ik * beta_i
for j = 1:D
  theta{j} = normalizzze(0.4+0.2*rand(nsyms(kids(j)), K));
  thetabeta{j} = zeros(K, N);
end

% note that 'xx' contains the beta messages (called lambda in Pearl)
ll = -inf;
change = 0;
iter = 0;
while (iter < maxiter) || ((best_ll < ll) && (best_ll > -Inf))
  iter = iter+1;
  % E-step combined with M-step (see note.pdf)
  %  calc q(z, x1, x2) = p(z, x1, x2 | observations)
  % first calculate the beta messages:
  beta = ones(K,N);
  for j = 1:D
    thetabeta{j} = theta{j}' * x{kids(j)};  % sum out the children
    beta = beta .* thetabeta{j};            % beta message for the latent
  end
  betapie = (pie*ones(1,N)) .* beta;
  sbetapie = sum(betapie, 1);
  % calculate loglikelihood
  old_ll = ll;
  ll = sum(log2(sbetapie), 2);
  %fprintf('[%s.m] ll==%f\n', mfilename, ll);
  if ll > 0
    error('[%s.m] bug detected: log-likelihood can not be positive', mfilename);
  end
  if old_ll - ll > 0
    if opt.verbose > 2
      fprintf('[%s.m] wrong direction: old-ll==%f\n', mfilename, old_ll-ll);
    end
  end
  if ll - old_ll < tol
    break
  end
  if opt.verbose > 2 %&& mod(iter, 100)==0
    fprintf('[%s.m] iter==%d  ll==%f change=%f\n', mfilename, iter, ll, change);
  end

  % continue with the combined E-step/M-step
  old = pie;
  betabetapie = betapie ./ (ones(K,1) * sbetapie);
  pie = normalizzze(sum(betabetapie, 2));
  change = 0;
  change = change + sum(abs(old(:)-pie(:)));
  for j = 1:D   % for all observed variables
    old = theta{j};
    % to avoid division by zero we need to set the zeros to one, 
    % this is correct since betabetapie should be also zero for those
    if 0
      nuller = find(thetabeta{j}(:)==0);
      if sum(abs(betabetapie(nuller))) > 0
        error('[%s] sanity check failed', mfilename);
      end
    end
    thetabeta{j}(thetabeta{j}(:)==0) = 1;
    betabetaj = betabetapie ./ thetabeta{j};   % remove kid from betabetapie
    theta{j} = normalizzze(theta{j} .* (x{kids(j)} * betabetaj'));
    change = change + sum(abs(old(:) - theta{j}(:)));
  end
end
%fprintf('[%s.m] done', mfilename);
CPTs.pz    = pie;
CPTs.px_z  = theta;
CPTs.qz_xi = beta;   % not a distribution but the beta/lambda messages
CPTs.df    = (K-1) + K*sum(nsyms(kids)-1);
