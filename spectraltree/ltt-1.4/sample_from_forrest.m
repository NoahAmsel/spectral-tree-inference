function data = sample_from_forrest(N, atree);
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

nvars = size(atree.t, 1);  % number of variables

t = atree.t;
p = atree.p;
t0 = atree.t0;
p0 = atree.p0;
% generate data from that tree distribution
x = zeros(nvars, N);
for n = 1:N
  for k = 1:length(t0) % loop over all roots
    root = t0(k);
    x(root, n) = sample(p0{k});   % sample a root
    stack = [root];
    while length(stack) > 0
      current = stack(1);
      stack = stack(2:end);
      for j = 1:size(t{current}, 2)  % loop over the kids of 'current'
        % sample a child dependent on its parent (=='current')
        x(t{current}(j), n) = sample(p{current}{j}(:, x(current, n)));
      end
      stack = [t{current}, stack];  % traverse depth-first
    end
  end
end

data.x = x;
data.nsyms = atree.nsyms;
data.name = atree.name;
