function t_hat = ltt(method, dtr, opt);
% wrapper for all other algorithms.
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

tic;
switch method
 case 'bin'
  t_hat = bin_forrest(dtr, opt);
 case 'cl'
  t_hat = chowliu(dtr);
 case 'ind'
  t_hat = ind_forrest(dtr);
 case 'lcm'
  t_hat = lcm_forrest(dtr);
 case 'zhang'
  t_hat = zhang_forrest(dtr);
 case 'hcs'
  opt.linkage = 'single';
  t_hat = hc_forrest(dtr, opt);
 case 'hca'
  opt.linkage = 'average';
  t_hat = hc_forrest(dtr, opt);
 case 'hcc'
  opt.linkage = 'complete';
  t_hat = hc_forrest(dtr, opt);
end
t_hat.time = toc;
