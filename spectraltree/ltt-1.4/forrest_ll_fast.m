function [l, missed] = forrest_ll(data, atree, verbose);
% calculuate the likelihood of data under a forrest model
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if isempty(atree)
  error('[%s.m] tree is empty', mfilename);
end
data.n = size(data.x, 2);

lls = 0;   % log-likelihood
ignored = 0;
for j = 1:length(atree.t0)   % loop over all roots
  msg = gen_message(atree.t0(j), data, atree);
  lls = lls + log(atree.p0{j}' * msg); % sum out the root variable
end
ignore = length(find(isinf(-lls)));
l = sum(lls(find(~isinf(-lls))));

if ignored > 0
  warning('[%s.m] %d of %d data points had zero prob.', mfilename, ...
                  ignored, size(data.x, 2));
end
missed = ignored;

function msg = gen_message(subtree, data, atree);
% the message is a likelihood of the leaves of the current subtree fixed
% given all values of the root of the current subtree
%
% e.g. for the tree   (x1 x2 (x3 x4 x5))
%
%          x1
%         /  \
%       x2    x3
%            /  \
%          x4    x5
%
%  p(<none> | x5)     = gen_message(5, ...);
%                     = (0 0 1 0 0);    % int x5 determining position of 1
%  p(<none> | x4)     = gen_message(4, ...);
%                     = (0 0 1 0 0);    % int x4 determining position of 1
%  p(x4, x5 | x3)     = gen_message(3, ...);
%                     = p(x4|x3) p(x5|x3)
%                     = (p(x4|x3)*p(<none>|x4)) .* (p(x5|x3)*p(<none>|x5))
%  p(<none> | x2)     = gen_message(2, ...);
%                     = (0 0 0 1 0);    % int x2 determining position of 1
%  p(x2, x4, x5 | x1) = gen_message(1, ...);
%                     = p(x2|x1)*sum_x3 p(x3|x1) p(x4,x5|x3)

% do we have data at the current node?
if subtree > atree.nobs
  % NOT OBSERVED
  % create a vector with ones that allows all values
  msg = ones(atree.nsyms(subtree), data.n);
else
  % OBSERVED
  % create a vector with zeros and a single one, which will pick out the
  % correct row from the CPT of the parent
  msg = zeros(atree.nsyms(subtree), data.n);
  for i=1:data.n
    if isfield(data, 'sparse') && data.sparse == 1
      msg(1+data.x(subtree,i), i) = 1;
    else
      msg(data.x(subtree,i), i) = 1;
    end
  end
end

% does subtree has kids?
nkids = size(atree.t{subtree}, 2);
if nkids > 0
  % ask for the messages of the kids
  for j = 1:nkids
    kid = atree.t{subtree}(j);
    kid_msg = gen_message(kid, data, atree);
    cpt = atree.p{subtree}{j};
    % the matrix product does sum out the kid
    msg = msg .* (cpt' * kid_msg);
  end
end
