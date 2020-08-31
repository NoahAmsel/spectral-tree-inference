function atree = lcm_forrest(d);
% generate an latent class model forrest, i.e. infer a single root
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

nvars = size(d.x, 1);
opt.N = size(d.x, 2);
[CPTs, ll] = em_automatic(data2distr(d), d.nsyms, 1:nvars, opt);
pz    = CPTs.pz;
px_z  = CPTs.px_z;
qz_xi = CPTs.qz_xi;
df = CPTs.df;
K = size(pz, 1);
root = nvars + 1;

atree.K = K;
atree.t0 = [root];
atree.p0 = cell(1,1);
atree.p0{1} = pz;
atree.t = cell(nvars+1, 1);
atree.t{root} = 1:nvars;
atree.p = cell(nvars+1, 1);
atree.p{root} = px_z;
atree.nsyms = [d.nsyms, K];
atree.nobs = nvars;
atree.name = [d.name '_lcm'];
atree.df = df;                 % degrees of freedom
check_forrest(atree);
