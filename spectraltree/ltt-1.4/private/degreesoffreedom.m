function df = degreesoffreedom(atree)
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

check_forrest(atree);
t0 = atree.t0;
t = atree.t;
nsyms = atree.nsyms;

% calculate degrees of freedom
df = 0;
for root = t0
  stack = [root];
  df = df + nsyms(root) - 1;
  while length(stack) > 0
    current = stack(1);
    stack = stack(2:end);
    for j = 1:size(t{current}, 2) % for all kids
      kid = t{current}(j);
      df = df + nsyms(current) * (nsyms(kid) - 1);
      stack = [kid, stack]; % traverse depth-first
    end
  end
end

