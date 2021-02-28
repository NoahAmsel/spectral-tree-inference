function atree = ind_forrest(d);
%generate an ind forrest
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

[nvars, nsamples] = size(d.x);
if nvars > nsamples
  warning('[%s.m] more variables than samples!', mfilename);
end

t0 = 1:nvars;
p0 = cell(1, nvars);
for i = 1:nvars
  rowi = d.x(i,:);   % all values of the i-th variable
  p0{i} = sample2dist(rowi, d.nsyms(i));
end

atree.t0 = t0;
atree.p0 = p0;
atree.t = cell(nvars, 1);
atree.p = cell(nvars, 1);
atree.nsyms = d.nsyms;
atree.nobs = nvars;
atree.name = [d.name '_ind'];
atree.df = degreesoffreedom(atree);
check_forrest(atree);