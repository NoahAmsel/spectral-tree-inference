function m = mi(pxy, signed)
% calcuate the mutual information between the two distribitions
% pxy contains the possibly unnormalizzzed countings corresponding to p(x,y)
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('signed', 'var') || isempty(signed)
  signed = 0;
end

% ensure normalization
pxy = pxy / sum(pxy(:));

% marginalize
px = sum(pxy, 1);     % p(x)
py = sum(pxy, 2);     % p(y)
pxpy = py*px;         % p(x) p(y)

rho = 0;
if signed == 1
  % change the sign of y if the variables are negatively correlated
  if all(size(pxy)==2)
    mx = px(2);   % mean x
    my = py(2);   % mean y
    sx = sqrt(px(1) * (0-mx)^2 + px(2) * (1-mx)^2);  % std x
    sy = sqrt(py(1) * (0-my)^2 + py(2) * (1-my)^2);  % std y
    cxy = pxy(1,1) * (0-mx)*(0-my) + ...
          pxy(1,2) * (1-mx)*(0-my) + ...
          pxy(2,1) * (0-mx)*(1-my) + ...
          pxy(2,2) * (1-mx)*(1-my);     % cross correlation
    rho = cxy / (sx*sy);                % correlation coefficient
  else
    warning('[mi.m] don''t use signed for non-binary variables');
  end
end

% we assume: 0*log2(0) == 0
% note that: pxpy(i,j)==0  ==>  pxy(i,j)==0
% sometimes this must be enforced (e.g. pxy(1,1)=1.e-214, but pxpy(1,1)=0)
% thus to avoid division by zero and log2(0.0) we need to:
pxy = pxy(:);
pxpy = pxpy(:);
pxy(pxpy==0) = 0;
pxpy(pxy==0) = 1;
pxy(pxy==0)  = 1;
m = pxy' * log2(pxy./pxpy);

if rho < 0
  m = -m;
end
