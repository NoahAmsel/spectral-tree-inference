function atree = hc_forrest(data, opt);
% use hierarchical clustering to estimate the tree and then learn the
% probabilities once the tree structure is given
% 
% inputs:
%    data
%    opt.linkage   what distance between clusters, possible values:
%                   'single'    min
%                   'complete'  max
%                   'average'   mean
%
% Copyright (C) 2006 - 2010 by Stefan Harmeling (2010-02-15).

if ~exist('opt', 'var')
  opt = []; 
end
if ~isfield(opt, 'verbose')
  opt.verbose = 0;
end
if all(data.nsyms==2), 
  opt.Kmax = 2; 
  fprintf('[%s.m] opt.Kmax=2;\n', mfilename);
end
if ~isfield(opt, 'signed')
  opt.signed = 0;
end
if opt.verbose > 1
  fprintf('[%s.m] chosen options (fields of struct "opt"):\n', mfilename);
  disp(opt);
end
if ~isfield(opt, 'linkage')
  opt.linkage = 'average';
end

% (i) calculate pairwise mutual information
if opt.verbose > 1
  fprintf('[%s.m] calculating mutual information...\n', mfilename)
end
m = mi_mat(data.x, data.nsyms, [], [], opt.verbose, opt.signed);
D = size(m, 1);
m(1:(D+1):end) = -Inf;     % ignore diagonal
if opt.verbose > 1
  m                          % print out the MI matrix
  fprintf('[%s.m] done\n', mfilename);
end

% (ii)  perform hierarchical clustering on the variables
t = hc(m, opt.linkage);

% (iii) estimate the probabilistic model
[xx, D, N] = data2distr(data);
% D == number of leaves
% N == number of data points
opt.N = N;
opt.D = D;
atree.t     = t;
atree.t0    = [length(t)];    % the last node in t is the single root
atree.xx    = xx;
atree.nsyms = data.nsyms;
atree = tree2model(atree, opt);
atree.name  = [data.name '_hc'];
switch opt.linkage
 case 'single',   atree.name = [atree.name 's'];
 case 'average',  atree.name = [atree.name 'a'];
 case 'complete', atree.name = [atree.name 'c'];
end
atree.df = degreesoffreedom(atree);

% refinement
if 1
  atree.ll_before_refine = forrest_ll_fast(data, atree);
  old_ll = -inf;
  new_ll = atree.ll_before_refine;
  counter = 1;
  while new_ll - old_ll >= 1e-2
    atree = forrest_refine(data, atree);
    old_ll = new_ll;
    new_ll = forrest_ll_fast(data, atree);
    counter = counter + 1;
    if counter > 100, break, end
  end
  atree.ll_after_refine = forrest_ll_fast(data, atree);
end


return
