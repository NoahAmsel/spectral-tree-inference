function ll = saturated_ll(d);
% calculate the "saturated log-likelihood"
%
% Given data points x1, x2, ..., xn, the likelihood of an 
% observation x under the saturated model is:
%
%         "number of times x appears in x1, ..., xn"
% p(x) = --------------------------------------------
%               "total number of data points"
%
% Stefan Harmeling, 5 feb 2008
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

f = sample2freq(d);

% we assume that the last column contains the frequencies
f = f(:, end);
N = sum(f);
ll = f' * log(f/N);
