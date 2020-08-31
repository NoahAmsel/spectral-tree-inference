function nsyms = samples2nsyms(x);
% checks whether x contains in each row only integers without holes
% and calculuates the number of symbols

msg = '[samples2nsyms.m] data does not fulfill all assumptions';
[nvar, N] = size(x);
nsyms = zeros(1, nvar);
for i = 1:nvar
  ux = unique(x(i, :));
  mux = max(ux);
  if length(ux) ~= mux
    error(msg);
  end
  if any(sort(ux)~=(1:mux))
    error(msg);
  end
  nsyms(i) = mux;
end
return
