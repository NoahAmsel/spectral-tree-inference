function m = mi_mat(x, nsyms, repeats, pvalue, verbose, signed)
% calculates a matrix containing all values of pair-wise mutual information
% for the distributions along the rows of x
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') || isempty(verbose)
  verbose = 0;
end
if ~exist('signed', 'var') || isempty(signed)
  signed = 0;
end

nvars = size(x, 1);
N = size(x, 2);
if exist('repeats', 'var') && ~isempty(repeats), 
  perm = 1; 
  xx = cell(nvars, 1);
  for i = 1:nvars
    xx{i} = zeros(nsyms(i), N);
    for n = 1:N
      xx{i}(x(i, n), n) = 1.0;
    end
  end
else 
  perm = 0; 
end

m = zeros(nvars, nvars);
for i = 1:nvars
  for j = i:nvars
    switch perm
     case 0
      d = sample2dist(x([i j], :), nsyms([i j]));
      m(i,j) = mi(d, signed);
     case 1
      m(i,j) = mi_perm(xx{i}, xx{j}, repeats, pvalue);
    end 
    if i < j
      m(j,i) = m(i,j);
    end
  end
end
