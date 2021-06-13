function atree = chowliu(data);
% Estimate the model proposed by Chow and Liu, 1968
%
% Stefan Harmeling, 28. JAN 2008
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('perm', 'var') || isempty(perm)
  perm = 0;
end

repeats = 1000;
pvalue = 0.99;



% (1) estimate the minimum spanning tree

% calculate the pairwise mutual informations
switch perm
 case 0
  m = mi_mat(data.x, data.nsyms); % pairwise mutual information
 case 1
  m = mi_mat(data.x, data.nsyms, repeats, pvalue); % pairwise mutual information
end
% calculate the minimum spanning tree using the negative mutual information
% as the weights to be minized
g = mst(-m);

% the tree can be arbitrarily be rooted, let's use 1
root = 1;

% (2) transform the matrix representation of the tree into the format with cell
% arrays
t0 = [root];  % a single root
t = cell(size(g, 1), 1);
stack = [root];   % start with node one
done = [];
while length(stack) > 0
  current = stack(1);
  stack = stack(2:end);
  kids = setdiff(find(g(current, :)), done);
  t{current} = kids;
  stack = [kids, stack];   % traverse depth-first
  done = [done current];
end

% (3) estimate the CPTs along the tree, i.e. for each edge estimate the
% two-variable distribution and then divide by the parent distribution
p0 = cell(1, 1);
p0{1} = sample2dist(data.x(root,:), data.nsyms(root));
stack = [root];  % start with node one
p = cell(size(t, 1), 1);
while length(stack) > 0
  current = stack(1);
  stack = stack(2:end);
  for j = 1:size(t{current}, 2)  % for all kids
    kid = t{current}(j);
    pair = [kid, current];
    pxy = sample2dist(data.x(pair,:), data.nsyms(pair));
    % calculate p(kid | current)
    py = sum(pxy, 1);    % sum out the kid
    valid = find(py ~= 0);
    p{current}{j} = zeros(size(pxy));
    p{current}{j}(:,valid) = pxy(:,valid) ./ repmat(py(valid), [size(pxy, 1) 1]);
    stack = [kid, stack];  % traverse depth-first
  end
end

atree.t = t;
atree.p = p;
atree.t0 = t0;
atree.p0 = p0;
atree.nsyms = data.nsyms;
atree.nobs = size(t, 1);    % all nodes are observed, no latent nodes
atree.name = [data.name '_cl'];
if perm == 1, atree.name = [atree.name 'p']; end
atree.df = degreesoffreedom(atree);
check_forrest(atree);
