function check_forrest(atree);
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

t = atree.t;
t0 = atree.t0;
p = atree.p;
p0 = atree.p0;
nsyms = atree.nsyms;
nobs = atree.nobs;

msg = sprintf('[%s.m] some checks failed!', mfilename);

% (0) check types and dimensions
if ~isnumeric(t0), error(msg); end
if ~iscell(p0), error(msg); end
if ~iscell(t), error(msg); end
if ~iscell(p), error(msg); end
nvars = size(t, 1);
nroots = size(t0, 2);
if size(t, 2) ~= 1, error(msg); end
if size(p, 2) ~= 1, error(msg); end
if nvars ~= size(p, 1), error(msg); end
% t0 and p0 are row vectors
if size(t0, 1) ~= 1, error(msg); end
if size(p0, 1) ~= 1, error(msg); end
if nroots ~= size(p0, 2), error(msg); end
if size(nsyms, 1) ~= 1, error(msg); end
if nvars ~= size(nsyms, 2), error(msg); end
if size(nobs, 1) ~= 1, error(msg); end
if size(nobs, 2) ~= 1, error(msg); end
if nobs > nvars, error(msg); end

% check whether 't' is a valid forrest
% (i) all numbers up to the maximum entries must appear exactly once
all = t0;  % start with the roots
nvars = size(t, 1);    % number of variables
for i=1:nvars
  for j=1:size(t{i}, 2)
    node = t{i}(j);
    all = [all, node];
  end
end
if sum(sort(all) ~= (1:nvars)) > 0, error(msg); end

% (ii) every node is reachable from the roots
reachable = [];
for i=1:nroots
  stack = [t0(i)];   % the roots
  while length(stack) > 0
    current = stack(1); stack = stack(2:end);
    if ismember(current, reachable)
      % cycle detected
      error(msg);
    end
    reachable = [reachable current];
    stack = [stack t{current}];   % add the kids
  end
end
if sum(sort(reachable) ~= (1:nvars)) > 0, error(msg); end

