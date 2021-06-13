function t_hat = evaluate(t_hat, dtr, dte);
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

[t_hat.lltr, t_hat.missedtr] = forrest_ll(dtr, t_hat);
[t_hat.llte, t_hat.missedte] = forrest_ll(dte, t_hat);
sat_ll = saturated_ll(dte);   % the saturated model
t_hat.lambda = 2 * (sat_ll - t_hat.llte);
t_hat.aic = aic(dte, t_hat, t_hat.llte);
t_hat.bic = bic(dte, t_hat, t_hat.llte);

if isfield(dtr, 'names')
  t_hat.names = dtr.names;
end