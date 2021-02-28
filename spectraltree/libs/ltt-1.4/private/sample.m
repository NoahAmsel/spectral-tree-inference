function x = sample(A);
% sample from distribution matrix A
% assume distributions along the first dimension
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

s = size(A);
A = A(:, :);
m = size(A, 1);
n = size(A, 2);
x = zeros(n, 1);
for i = 1:n
  p = cumsum(A(:, i));
  x(i) = min(find(rand*ones(m, 1) < p));
end
if length(s) > 2
  x = reshape(x, s(2:end));
end