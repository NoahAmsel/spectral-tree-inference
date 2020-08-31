% ltt_script.m
% demonstrates how to call the algorithms in this toolbox.
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

clear all;
close all;

addpath ./harmeling/ltt-1.3/


savefile = 1;
dataset = 13;
method = 'bin';

if(dataset==13)
    load newsgroup
    RESULTS_PATH = 'results_newsgroup3';
    dtr.name = 'newsgroup';
    nodeLabels = wordlist;
    trainSamples = samples;
    clear samples testSamples;
elseif(dataset==100)
    load sunObjects;
    RESULTS_PATH = 'results_sun';
    dtr.name = 'objects';
    nodeLabels = names;
end
dtr.x = trainSamples;
dtr.nsyms = 2*ones(1,size(trainSamples,1));
%dte.x = testSamples;
%dte.nsyms = dtr.nsyms;

% (2) call method
% 'bin'    binary trees (the proposed method in our paper)
% 'ind'    independent leaves
% 'lcm'    latent class model (single parent for all variables)
% 'cl'     Chow-Liu's trees
% 'zhang'  Method of Nevin Zhang, see refs.
if ~exist('method', 'var')
  method = 'bin';
end
fprintf('[%s.m] running method "%s" on dataset %d "%s"\n', mfilename, method, dataset, dtr.name);
if ~exist('opt', 'var')
  opt = [];
end
opt.verbose = 0;
t_hat = ltt(method, dtr, opt);
  
newNodeLabels = cell(length(t_hat.t),1);
for i=1:length(t_hat.t)
    if(i <= t_hat.nobs)
        newNodeLabels{i} = nodeLabels{i};
    else
        newNodeLabels{i} = ['h' num2str(i-t_hat.nobs)];
    end
end

t_hat.nodeLabels = newNodeLabels;

% (3) show some results
[t_hat.lltr, t_hat.missedtr] = forrest_ll(dtr, t_hat);
t_hat.bictr = t_hat.lltr - 0.5*t_hat.df*log(size(trainSamples,2));
%[t_hat.llte, t_hat.missedte] = forrest_ll(dte, t_hat);
%t_hat.bicte = t_hat.llte - 0.5*t_hat.df*log(size(testSamples,2));

fprintf('running "%s" on "%s"\n', method, dtr.name);
fprintf('------------------------------------\n');
fprintf('time == %f\n', t_hat.time);
if isfield(t_hat,'tStructure')
    fprintf('time to learn structure == %f\n', t_hat.timeStructure);
    fprintf('time to fit parameters using EM == %f\n', t_hat.timeParameter);
end
fprintf('number of hidden variables == %d\n', length(t_hat.t)-t_hat.nobs);
fprintf('ll (train)  == %f\n', t_hat.lltr);
fprintf('BIC (train)  == %f\n', t_hat.bictr);
%fprintf('ll (test)  == %f\n', t_hat.llte);
%fprintf('BIC (test)  == %f\n', t_hat.bicte);
%fprintf('%.0f %.0f %.0f %.0f %d %d %.1f\n', t_hat.lltr, t_hat.bictr, t_hat.llte, t_hat.bicte, length(t_hat.t)-t_hat.nobs, t_hat.df, t_hat.time);
fprintf('%.0f %.0f %d %d %.1f\n', t_hat.lltr, t_hat.bictr, length(t_hat.t)-t_hat.nobs, t_hat.df, t_hat.time);
fprintf('------------------------------------\n');

% save final results
clear samples dte dtr trainSamples testSamples
if(savefile)
    fname = sprintf('%s/%s_%s', RESULTS_PATH, t_hat.name, date);
    fprintf('saving final results in "%s.mat"\n', fname);
    save(fname);
end
