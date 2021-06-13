clear all;
close all;

load ./data/newsgroup
addpath('./toolbox');

trainSamples = samples;
clear samples
%clear samples testSamples;
options.nodeLabels = wordlist;

savefile = 0;
options.root = 1;
options.verbose = 1;
options.maxHidden = inf;
options.startingTree = 'MST';

m = 32;
K = 2;
pr_bounds = [0.9 0.95];
n = 10^4;
% symmetric tree
M = reshape([eye(m-1) ; eye(m-1)],m-1,2*(m-1))';
% generate tree data
T = generate_transition_matrices(K,m,pr_bounds,1);
        
% Sample data according to confusion matrix
%X = generate_tree_samples_multiclass(M,n,T,(1/K)*ones(1,K));        


method = 'RG';
% method = 'NJ';
%method = 'CLRG';
% method = 'CLNJ';
% method = 'regCLRG';
% method = 'regCLNJ';

tic;
switch method
    case 'RG'
        [adjmatT,optionsEM.edge_distance] = RG(trainSamples, 0);
    case 'NJ'
        [adjmatT,optionsEM.edge_distance] = NJ(trainSamples, 0);
        %[adjmatT,optionsEM.edge_distance] = NJ(X, 0);
    case 'CLRG'
        [adjmatT,optionsEM.edge_distance] = CLRG(trainSamples, 0);
    case 'CLNJ'
        [adjmatT,optionsEM.edge_distance] = CLNJ(trainSamples, 0);
    case 'regCLRG'
        [adjmatT, edgeD, optionsEM.initNodePot, optionsEM.initEdgePot, ll, bic] = regCLRG_discrete(trainSamples, options);       
    case 'regCLNJ'
        [adjmatT, edgeD, optionsEM.initNodePot, optionsEM.initEdgePot, ll, bic] = regCLNJ_discrete(trainSamples, options);
        
end
tStructure = toc;

tic;
optionsEM.root = options.root;
if(~isfield(optionsEM,'initNodePot'))
    optionsEM.max_ite = 10;
    optionsEM.numStarting = 5;
    for i=1:optionsEM.numStarting
        [node_pot_int{i}, edge_pot_int{i}, ll_approx] = learnParamsEMmex(trainSamples,adjmatT,optionsEM);
        ll_em_int(i) = ll_approx(end);
    end
    [foo,ind] = max(ll_em_int);
    optionsEM.initNodePot = node_pot_int{ind};
    optionsEM.initEdgePot = edge_pot_int{ind};
    clear node_pot_int edge_pot_int
end
optionsEM.max_ite = 300;
[nodePot, edgePot, ll_approx] = learnParamsEMmex(trainSamples,adjmatT,optionsEM);
tParameter = toc;


%  This evaluation part is modified from the code by Stefan Harmeling 
%  which can be downloaded from 
%  http://people.kyb.tuebingen.mpg.de/harmeling/code/ltt-1.3.tar
if(exist('testSamples','var'))
    t_hat = tree2Harmeling(adjmatT,options.root,nodePot,edgePot,method,tStructure,tParameter,options.nodeLabels,trainSamples,testSamples);
else
    t_hat = tree2Harmeling(adjmatT,options.root,nodePot,edgePot,method,tStructure,tParameter,options.nodeLabels,trainSamples);
end

fprintf('running "%s"\n', method);
fprintf('------------------------------------\n');
fprintf('time == %f\n', t_hat.time);
if isfield(t_hat,'timeStructure')
    fprintf('time to learn structure == %f\n', t_hat.timeStructure);
    fprintf('time to fit parameters using EM == %f\n', t_hat.timeParameter);
end
fprintf('number of hidden variables == %d\n', length(t_hat.t)-t_hat.nobs);
fprintf('ll (train)  == %f\n', t_hat.lltr);
fprintf('BIC (train)  == %f\n', t_hat.bictr);
if(exist('testSamples','var'))
    fprintf('ll (test)  == %f\n', t_hat.llte);
    fprintf('BIC (test)  == %f\n', t_hat.bicte);
end
fprintf('------------------------------------\n');

% save final results
clear trainSamples testSamples
if(savefile)
    fname = sprintf('results/%s', t_hat.name);
    fprintf('saving final results in "%s.mat"\n', fname);
    save(fname);
end