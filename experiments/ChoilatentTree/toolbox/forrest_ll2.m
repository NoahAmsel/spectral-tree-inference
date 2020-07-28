function [l, missed] = forrest_ll2(data, atree, verbose);
% calculuate the log-likelihood of data under a forrest model
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') || isempty(verbose)
  verbose = 0;
end

if isempty(atree)
  error('[%s.m] tree is empty', mfilename);
end

l = 0;   % log-likelihood
ignored = 0;
min_ll_datum = inf;
%%%fprintf('[%s.m] calculating the log-likelihood of the data\n');
for i = 1:size(data.x, 2)
  if verbose > 1
    fprintf('[%s.m] %d/%d\n', mfilename, i, size(data.x, 2));
  end
  datum = data.x(:, i);
  ll_datum = 0;
  for j = 1:length(atree.t0)   % loop over all roots
    msg = gen_message(atree.t0(j), datum, atree);
    ll_datum = ll_datum + log(msg * atree.p0{j}); % sum out the root variable
  end
  if ll_datum == -Inf
    ignored = ignored + 1;
  else     
      min_ll_datum = min(min_ll_datum,ll_datum);
    l = l + ll_datum;
  end
end
if ignored > 0
  warning(sprintf('[%s.m] %d of %d data points had zero prob.', mfilename, ...
                  ignored, size(data.x, 2)));
  fprintf('Using the minimum lilkelihood %f instead of -Inf\n',min_ll_datum);
              l = l+ignored*min_ll_datum;
end
missed = ignored;

function msg = gen_message(subtree, datum, atree);
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
  msg = ones(1, atree.nsyms(subtree));
else
  % OBSERVED
  % create a vector with zeros and a single one, which will pick out the
  % correct row from the CPT of the parent
  msg = zeros(1, atree.nsyms(subtree));
  msg(datum(subtree)) = 1;
end

% does subtree has kids?
nkids = size(atree.t{subtree}, 2);
if nkids > 0
  % ask for the messages of the kids
  for j = 1:nkids
    kid = atree.t{subtree}(j);
    kid_msg = gen_message(kid, datum, atree);
    cpt = atree.p{subtree}{j};
    % the matrix product does sum out the kid
    msg = msg .* (kid_msg*cpt);
  end
end
