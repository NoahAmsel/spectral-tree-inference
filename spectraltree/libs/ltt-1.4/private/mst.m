function m = mst(w);
%Calculates the minimum spanning tree.
% 
% input:
%      w   matrix of weights
%
% output:
%      m   a binary matrix denoting the edges
%     
% Stefan Harmeling, 28. JAN 2008 (reusing older code)
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

[n, nn] = size(w);
if n ~= nn
  error('[mst.m] squared matrix expected');
end
t = zeros(n, n);      % initially no edges

% there are no potential edges on the diagonal
w(1:(n+1):end) = Inf;

% at the beginning only vertix 1 is in the tree
tree = 1;

% 'keys' contains the shortest distances to the tree
keys = w(:,1);

% 'parents' contains the nodes corresponding to the 'keys' in the tree
parents = ones(n, 1);

% the tree has exactly n-1 edges
for count = 1:(n-1)

  % ignore the tree members
  keys(tree) = Inf;
  
  % who is the 'next' member of the tree
  [dummy, next] = min(keys);

  % connect 'next' to the tree
  tree = [tree next];
  m(next, parents(next)) = 1;
  m(parents(next), next) = 1;
  
  % update the keys
  [keys, ind] = min([keys, w(:, next)], [], 2);
  
  % update parents
  parents(find(ind==2)) = next;
end

