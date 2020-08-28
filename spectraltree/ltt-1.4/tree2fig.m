function tree2fig(t);
%Plots a tree into a figure.
%
% Copyright (C) 2006-2009 by Stefan Harmeling.

roots = t.t0;                  % the list of roots
children = t.t;                % the list of children for each node
nsyms = t.nsyms;
n = length(children);          % number of nodes
                               
parents = zeros(1,n);          % a list of parents
stack = roots;                 % which nodes need to be processed
while ~isempty(stack)         
  node = stack(1);            
  stack = stack(2:end);        % pop off an element
  kids = children{node};
  for kid = kids
    parents(kid) = node;
  end
  stack = [stack, kids];       % push the children of node
end

[x,y] = treelayout(parents, 1:n);

% draw lines
cla
stack = roots;
while ~isempty(stack)
  node = stack(1);
  stack = stack(2:end);        % pop off an element
  kids = children{node};
  for kid = kids
    line([x(node), x(kid)], [y(node), y(kid)]);
  end
  stack = [stack, kids];   % push the children
end

% draw labels
stack = roots;
while ~isempty(stack)
  node = stack(1);
  stack = stack(2:end);        % pop off an element
  tt = text(x(node), y(node), sprintf('%d(%d)', node, nsyms(node)));
  set(tt, 'HorizontalAlignment', 'center');
  set(tt, 'VerticalAlignment', 'middle');
  set(tt, 'FontSize', 12)
  kids = children{node};
  stack = [stack, kids];   % push the children of node
end

% set up figure
set(gca, 'XLim', [0, 1])
set(gca, 'YLim', [0, 1])
axis equal
axis off
