function p = sample2dist(x, nsyms);
% Creates the empirical distribution for a matrix of samples.
% Each column is assumed to be one example.
% Each row corresponds to a random variable.
% The values of x(i,:) are assumed to come from 1:nsyms(i).
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

lnsyms = length(nsyms);
[nvar, nsamples] = size(x);
if lnsyms ~= nvar
  error('[sample2dist.m] x and nsyms missmatch');
end

if lnsyms == 1
  p = zeros(nsyms, 1);
  for i = 1:nsyms
    p(i) = sum(x==i);
  end
elseif lnsyms == 2
  p = zeros(nsyms);
  for i = 1:nsyms(1)
    for j = 1:nsyms(2)
      p(i,j) = sum(x(1,:)==i & x(2,:)==j);
    end
  end
else
  error('[sample2dist.m] x can have only one or two rows');
end
p = p/nsamples;
