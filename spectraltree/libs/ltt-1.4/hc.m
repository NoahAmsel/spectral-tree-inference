function t = hc(d, link);
%HC performs hierarchical clustering.
%
% inputs:
%   d     a distance matrix
%   link  one of 'single', 'complete', 'average'
%
% outputs:
%   t     a cell array containing the kids of a node, 
%         e.g. t{15} == [3,4,5]  nodes 3,4,5 are the kids of node 15
%
% note:
%   the root is always the last node in t.  this structure is a bit
%   wasteful, since the first nleaves entries are empty.
%
% Stefan Harmeling * 16-FEB-2010

nleaves = size(d, 1);
nnodes  = 2*nleaves - 1;    % total number of nodes

if size(d, 1) ~= size(d, 2)
  error('[%s.m] square matrix expected', mfilename);
end

msg = sprintf('[%s.m] bug detected', mfilename);
d(1:(nleaves+1):end) = -Inf;     % ignore the diagonal
frontier = 1:nleaves;
csizes = ones(1, nnodes);       % number of leaves below node
next = nleaves + 1;              % the index of the next inner node
t = cell(nnodes, 1);             % cell array with lists of children
lfrontier = length(frontier);
while lfrontier > 1
  dsel = d(frontier, frontier);
  [dummy, idx] = max(dsel(:));
  if dummy == -Inf
    error(msg);
  end
  i = mod(idx-1, lfrontier) + 1;   % row index of maximum
  j = ceil(idx/lfrontier);         % col index of maximum
  if i==j
    error(msg);
  end
  i = frontier(i);             % remap
  j = frontier(j);             % remap
  t{next} = [i,j];             % connect the new node with its children
  d(next, :) = NaN;            % initialize new entries
  d(:, next) = NaN;
  frontier(frontier==i) = [];  % remove kids from the frontier
  frontier(frontier==j) = [];  % remove kids from the frontier
  csizes(next) = csizes(i) + csizes(j);
  for f = frontier
    switch link
     case 'single'
      d(next, f) = min(d(i, f), d(j, f));
     case 'complete'
      d(next, f) = max(d(i, f), d(j, f));
     case 'average'
      d(next, f) = (csizes(i)*d(i, f) + csizes(j)*d(j, f))/csizes(next);
    end
    d(f, next) = d(next, f);
  end
  frontier(end+1) = next;      % add new latent to the frontier
  lfrontier = length(frontier);
  next = next + 1;
end
% sanity check:
% each number from 1 to nnodes-1 should appear exactly once in t
if any(sort([t{:}]) ~= 1:(nnodes-1))
  error('[%s.m] sanity check failed', mfilename);
end
return
