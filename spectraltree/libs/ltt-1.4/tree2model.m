function atree = tree2model(atree, opt);
%
% inputs:
%   atree   a tree structure with .t, .t0, .nsyms, .xx
%   opt     a struct of options
%
% outputs:
%   atree   a tree structure with probability distributions at each latent node
%
% note that if latent nodes have only a single state, they become
% roots.  thus the resulting tree might have more roots than the
% initial one.
%
% Copyright (C) 2006 - 2010 by Stefan Harmeling (2010-02-18).

D = length(atree.xx);      % number of observations
N = size(atree.xx{1}, 2);  % number of data points
atree.p0 = {};             % the distributions of the roots
atree.p  = cell(D, 1);     % the conditional probability distributions
atree.betas = atree.xx;    % the initial beta messages are just the xx
atree.nobs  = D;

% go recursively down to the leaves and start there up until the roots
t0 = atree.t0;
for i = 1:length(t0)   % loop over all roots
  root = t0(i);
  atree = estmodel(atree, root, opt);
end

% prune away not needed nodes (i.e. those with nsyms == 1) for this we first
% create a mapping which maps latent nodes to their new location
nsyms  = atree.nsyms;
t      = atree.t;
p      = atree.p;
p0     = atree.p0;
nobs   = atree.nobs;       % number of leaves
xx     = atree.xx;
betas  = atree.betas;
lnsyms = length(nsyms);    % total number of nodes
map    = 1:lnsyms;         % initialize with identity
next   = nobs+1;           % next index for a latent node
t0     = [];               % a new list of roots
for i = (nobs+1):lnsyms
  if nsyms(i) > 1
    map(i) = next;
    next   = next + 1;
  else
    for kid = t{i}
      if nsyms(kid) > 1
        t0     = [t0, kid];   % add the kid as a new root
      end
    end
    map(i) = 0;            % map to nowhere
  end
end
% map the list of roots
if isempty(t0)
  % special case: complete tree
  t0 = lnsyms;
else
  % map each root
  for i = 1:length(t0)
    t0(i) = map(t0(i));
  end
end
  
% having the map we can move and rename all nodes in t %
% note that the first latent node (nobs+1) can be left untouched since
% either it stays as it is, or it will be overwritten later
for i = (nobs+2):length(nsyms)
  kids = [];
  for kid = t{i}
    kids(end+1) = map(kid);
  end
  if map(i) > 0
    nsyms(map(i)) = nsyms(i);
    t{map(i)}     = kids;
    p{map(i)}     = p{i};
    xx{map(i)}    = xx{i};
    betas{map(i)} = betas{i};
  end
end
% remove the not needed stuff
sel = next:lnsyms;
nsyms(sel) = [];
t(sel) = [];
p(sel) = [];
xx(sel) = [];
betas(sel) = [];

% estimate the distributions of the roots
p0 = cell(1, length(t0));
for i = 1:length(t0)
  root = t0(i);
  p0{i} = sum(xx{root}, 2)/N;
end
atree.nsyms = nsyms;
atree.t0    = t0;
atree.t     = t;
atree.p0    = p0;
atree.p     = p;
atree.xx    = xx;
atree.betas = betas;
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function atree = estmodel(atree, node, opt);

% the recursion
stop_here = 0;
kids  = atree.t{node};
for kid = kids
  % learn a model for the kid (the recursion)
  atree = estmodel(atree, kid, opt);
  if atree.nsyms(kid) < 2, 
    % if one of the (two) kids has only a single state we stop
    stop_here = 1;
  end
end

% check whether we learned something
if stop_here || node <= atree.nobs
  % stopping condition of the recursion
  % this happens for leaves, since they don't have kids and for latent
  % nodes for which the kids also only have single symbols
  if length(kids) > 0
    % we are not at a leaf
    atree.nsyms(node) = 1;
    atree.p{node} = [];
    atree.xx{node} = [];
    atree.betas{node} = [];
  end
else
  nobs  = atree.nobs;        % number of observations
  nsyms = atree.nsyms;
  betas = atree.betas;
  xx    = atree.xx;
  p     = atree.p;
  % next learn the model at this node
  if opt.verbose > 0
    fprintf('[%s.m] candidate nodes %d and %d\n', mfilename, kids(1), kids(2));
  end
  [CPTs, ll, K, opt] = em_automatic(betas, nsyms, kids, opt);
  beta_z = CPTs.qz_xi;  % the beta messsage for the new node
                        % qz_xi can be calculated from beta_z
  pie = CPTs.pz;
  qz_xi = beta_z .* (pie * ones(1, size(beta_z, 2)));
  qz_xi = qz_xi ./ (ones(K, 1) * pie' * beta_z);
  betas{node} = beta_z;
  px_z  = CPTs.px_z;
  
  xx{node} = qz_xi;
  if K~=size(qz_xi, 1)
    error('[%s.m] bug!', mfilename);
  end
  p{node} = px_z;
  nsyms(node) = K;
  if opt.verbose > 0
    if K > 1
      fprintf('[%s.m] add link %d -> %d and %d -> %d\n', mfilename, node, kids(1), node, kids(2));
    else
      fprintf('[%s.m] no kids for %d\n', mfilename, node);
    end
  end
  atree.xx    = xx;
  atree.nsyms = nsyms;
  atree.betas = betas;
  atree.p     = p;
end
return
