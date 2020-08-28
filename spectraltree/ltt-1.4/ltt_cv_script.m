% ltt_cv_script.m
% demonstrates how to call the algorithms in this toolbox.
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

% (1) data
% 1  binary-forest
% 2  three-level-binary
% 3  three-coins
% 4  six-coins
% 5  coleman
% 6  hiv-test
% 7  house-building
% 8  hannover-5
% 9  hannover-8
% 10 uci car evaluation
% 11 vision
% 12 vision01
% 13 news20w100 (not included, can be downloaded from http://www.cs.toronto.edu/~roweis/data/20news_w100.mat
% 14 coil-86
% 15 coil-42 (from Zhang)
dataset = 13;
[dtr, dte, t] = create_data(dataset);

% (2) pick method
% 'bin'    binary trees (the proposed method in our paper)
% 'ind'    independent leaves
% 'lcm'    latent class model (single parent for all variables)
% 'cl'     Chow-Liu's trees
% 'zhang'  Method of Nevin Zhang, see refs.
% 'hcs'    hierarchical clustering with single linkage
% 'hca'    hierarchical clustering with average linkage
% 'hcc'    hierarchical clustering with complete linkage
method = 'hcs';

% (3) run cross validation
cv = 10;                          % number of folds for x-fold cross validation
if ~isempty(t), 
  t = evaluate(t, dtr, dte);        % generate some statistics
end
ntr = size(dtr.x, 2);             % how many training examples
x_all = dtr.x(:, randperm(ntr));  % x_all stores all points
nholdout = floor(ntr/cv);         % how many examples are holdout during cv
cv_acc = zeros(cv, 1);            % accuracy for each holdout
cv_aic = zeros(cv, 1);            % AIC for each holdout
cv_bic = zeros(cv, 1);            % BIC for each holdout
cv_time = zeros(cv, 1);           % all times for each run
opt.restarts = 10;                % number EM restarts
opt.tol = 1e-8;                   % convergence tolerance for EM (lcm_lambda.m)
opt.binarysearch = 0;             % do a binary search in em_automatic.m
opt.verbose = 0;
fprintf('[%s.m] running method "%s" on dataset %d "%s"\n', mfilename, method, dataset, dtr.name);
for fold = 1:cv
  if cv > 1
    fprintf('[%s.m] CROSS-VALIDATION fold %d of %d\n', mfilename, fold, cv);
    if fold < cv
      sel = (1:nholdout) + (fold-1)*nholdout;
      dte.x = x_all(:, sel);
      dtr.x = x_all(:, setdiff(1:ntr, sel));
    else
      % last round make sure all points are used
      dte.x = x_all(:, (nholdout*(cv-1)+1):end);
      dtr.x = x_all(:, 1:(nholdout*(cv-1)));
    end
  end
  
  t_hat = ltt(method, dtr, opt);    % call algorithm

  if cv > 1
    cv_acc(fold) = forrest_ll(dte, t_hat);
    cv_aic(fold) = aic(dte, t_hat);
    cv_bic(fold) = bic(dte, t_hat);
    cv_time(fold) = t_hat.time;
  end
end
t_hat.cv_acc = cv_acc;
t_hat.cv_aic = cv_aic;
t_hat.cv_bic = cv_bic;
t_hat.cv_time = cv_time;


% (4) show some results
t_hat = evaluate(t_hat, dtr, dte);
fprintf('[%s.m] ------------------------------------\n', mfilename);
fprintf('[%s.m]  running "%s" on "%s"\n', mfilename, method, dtr.name);
fprintf('[%s.m]  %d-fold cross-validated\n', mfilename, cv);
fprintf('[%s.m]  shown are means of the folds\n', mfilename);
fprintf('[%s.m] ------------------------------------\n', mfilename);
fprintf('[%s.m]  time == %f +- %f\n', mfilename, mean(t_hat.cv_time), std(t_hat.cv_time));
fprintf('[%s.m]  ll (test) == %f +- %f\n', mfilename, mean(t_hat.cv_acc),  std(t_hat.cv_acc));
fprintf('[%s.m]  AIC (test) == %f +- %f\n', mfilename, mean(t_hat.cv_aic),  std(t_hat.cv_aic));
fprintf('[%s.m]  BIC (test) == %f +- %f\n', mfilename, mean(t_hat.cv_bic),  std(t_hat.cv_bic));
fprintf('[%s.m] ------------------------------------\n', mfilename);

% save final results
RESULTS_PATH = 'results';
if ~exist(RESULTS_PATH, 'dir'); mkdir(RESULTS_PATH); end
fname = sprintf('%s/%s_%s', RESULTS_PATH, t_hat.name, date);
fprintf('[%s.m] saving final results in "%s.mat"\n', mfilename, fname);
save(fname);

% plot the tree
tree2fig(t_hat)       % pure matlab
%%%tree2dot(t_hat, fname, 'png'); % might work but requires GraphViz
