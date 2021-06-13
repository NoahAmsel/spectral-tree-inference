function atree = orient(atree, verbose);
% ORIENT tries to orient the binary values of the latent variables, such
% that the occurence of twos is sparse.
%
% Note that for variables with two values, 
%    zero   ->  1
%    one    ->  2
% thus twos should be sparse.
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') || isempty(verbose)
  verbose = 0;
end

if any(atree.nsyms(:)~=2)
  warning('[%s.m] some nodes are NOT binary', mfilename);
end

if ~isfield(atree, 'pm'), atree = marginals(atree); end

% Run recursively down to the leaves and start there up until the root.
for i = 1:length(atree.t0)   % loop over all roots
  root = atree.t0(i);
  p = atree.pm{root};
  if root > atree.nobs   % don't flip leaves/observables
    flipped = 0;
    if p(1) < p(end)
      % do the flip
      if verbose > 1
        fprintf('[%s.m] flipud(atree.p0{%d})\n', mfilename, i);
      end
      atree.p0{i} = flipud(atree.p0{i});
      atree.pm{root} = flipud(atree.pm{root});
      flipped = 1;
    end
    atree = orient_topdown(flipped, atree, root, verbose);
  end
end

function atree = orient_topdown(flipped, atree, subtree, verbose)
nkids = length(atree.t{subtree});
if verbose > 1
  fprintf('[%s.m] orient_topdown(flipped=%d, subtree=%d, nkids=%d)\n', mfilename, flipped, subtree, nkids);
end
if flipped
  % flip the conditional variable in the CPTs of the kids
  for j = 1:nkids
    if verbose> 1
      fprintf('[%s.m] fliplr(atree.p{%d}{%d})\n', mfilename, subtree, j); 
    end
    atree.p{subtree}{j} = fliplr(atree.p{subtree}{j});
  end
end
% next check the children whether they flip as well
for j = 1:nkids
  kid = atree.t{subtree}(j);
  p = atree.p{subtree}{j} * atree.pm{subtree};
  if kid > atree.nobs  % don't flip leaves/observables
    flipped = 0;
    if p(1) < p(end)
      if verbose > 1
        fprintf('[%s.m] flipud(atree.p{%d}{%d})\n', mfilename, subtree, j); 
      end
      atree.p{subtree}{j} = flipud(atree.p{subtree}{j});
      atree.pm{kid} = flipud(atree.pm{kid});
      flipped = 1;
    end
    atree = orient_topdown(flipped, atree, kid, verbose);
  end
end
