function atree = marginals(atree, verbose)
% calculates marginals for each node.
% 21jan2009
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') || isempty(verbose)
  verbose = 0;
end

% store all marginal distribution in .pm
atree.pm = cell(length(atree.nsyms), 1);

% Run recursively down to the leaves and start there up until the root.
for i = 1:length(atree.t0)   % loop over all roots
  root = atree.t0(i);
  p = atree.p0{i};
  if root > atree.nobs   % are there any children?
    atree = marginals_topdown(atree, root, p, verbose);
  end
  atree.pm{root} = p;  % store marginal distribution
end

function atree = marginals_topdown(atree, subtree, pp, verbose)
nkids = length(atree.t{subtree});
if verbose > 1
  fprintf('[%s.m] marginals_topdown(subtree=%d, nkids=%d)\n', mfilename, subtree, nkids);
end
for j = 1:nkids
  kid = atree.t{subtree}(j);
  p = atree.p{subtree}{j} * pp;
  if kid > atree.nobs  % don't flip leaves/observables
    atree = marginals_topdown(atree, kid, p, verbose);
  end
  atree.pm{kid} = p;  % store marginal distribution
end
