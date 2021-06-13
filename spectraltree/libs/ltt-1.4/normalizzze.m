function a = normalizzze(a);
%NORMALIZZZE normalizes columns.
%
% Stefan Harmeling
a = a ./ repmat(sum(a,1), [size(a,1),1]);
return