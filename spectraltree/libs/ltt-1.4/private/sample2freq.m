function f = sample2freq(d);
% convert a data table with nsyms to a frequency table, where each example
% is a row and the last column contains the frequencies
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

x = d.x;
N = size(x, 2);
u = unique(x', 'rows')';

% loop over all unique column vectors to count their frequencies
f = zeros(size(u, 2), size(x, 1) + 1);   % last column for the frequencies
for i=1:size(u, 2);
  % number of matching vectors
  lmatching = sum(~sum(abs(x - repmat(u(:,i), [1 N])), 1));
  f(i, 1:end-1) = u(:,i)';
  f(i, end) = lmatching;
end
if sum(f(:, end)) ~= N
  error('[sample2freq.m] bug detected!');
end
