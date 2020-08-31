% ltt_script.m
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
if ~exist('dataset', 'var'), dataset = 11; end
[dtr, dte] = create_data(dataset);

% (2) call method
% 'bin'    binary trees (the proposed method in our paper)
% 'ind'    independent leaves
% 'lcm'    latent class model (single parent for all variables)
% 'cl'     Chow-Liu's trees
% 'zhang'  Method of Nevin Zhang, see refs.
% 'hcs'    hierarchical clustering with single linkage
% 'hca'    hierarchical clustering with average linkage
% 'hcc'    hierarchical clustering with complete linkage
if ~exist('method', 'var'), method = 'bin'; end

fprintf('[%s.m] running method "%s" on dataset %d "%s"\n', mfilename, method, dataset, dtr.name);
if ~exist('opt', 'var')
  opt = [];
end
opt.verbose = 2;
t_hat = ltt(method, dtr, opt);
  
% save intermediate results
RESULTS_PATH = 'results';
if ~exist(RESULTS_PATH, 'dir'); mkdir(RESULTS_PATH); end
fname = sprintf('%s/%s_%s', RESULTS_PATH, t_hat.name, date);
fprintf('[%s.m] saving intermediate results in "%s.mat"\n', mfilename, fname);
save(fname);

% (3) show some results
t_hat = evaluate(t_hat, dtr, dte);
fprintf('[%s.m] ------------------------------------\n', mfilename);
fprintf('[%s.m]  running "%s" on "%s"\n', mfilename, method, dtr.name);
fprintf('[%s.m] ------------------------------------\n', mfilename);
fprintf('[%s.m]  time == %f\n', mfilename, t_hat.time);
fprintf('[%s.m]  ll (train)  == %f\n', mfilename, t_hat.lltr);
fprintf('[%s.m]  ll (test)   == %f\n', mfilename, t_hat.llte);
fprintf('[%s.m]  AIC (test)  == %f\n', mfilename, t_hat.aic);
fprintf('[%s.m]  BIC (test)  == %f\n', mfilename, t_hat.bic);
fprintf('[%s.m] ------------------------------------\n', mfilename);

% plot the tree
tree2fig(t_hat)       % pure matlab
%tree2dot(t_hat, fname, 'pdf'); % might work but requires GraphViz
%tree2dot(t_hat, fname, 'png'); % might work but requires GraphViz
%tree2dot(t_hat, fname, 'svg'); % might work but requires GraphViz

% save final results
fprintf('[%s.m] saving final results in "%s.mat"\n', mfilename, fname);
save(fname);
