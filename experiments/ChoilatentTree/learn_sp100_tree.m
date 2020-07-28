close all
clear all
addpath ./toolbox
load ./data/cov_sp100
sample_cov = cov_mat;
%sample_cov = cov_mat(1:end-1,1:end-1);
%ticker = ticker(1:end-1);
root = 85;
m = size(sample_cov,1);
D = diag(1./sqrt(diag(sample_cov)));
rho_mat = D*sample_cov*D;

distance = -log(abs(rho_mat));

useDistances = 1;
verbose = 0;
numSamples = 216;
comp_time = zeros(5,1);

tic;
adjmatT{1} = ChowLiu(-distance);
edgeD{1} = distance.*adjmatT{1};
comp_time(1) = toc; tic;
[adjmatT{2},edgeD{2}] = CLNJ(distance, useDistances);
comp_time(2) = toc; tic;
[adjmatT{3},edgeD{3}] = CLRG(distance, useDistances, numSamples);
comp_time(3) = toc; tic;
[adjmatT{4},edgeD{4}] = NJ(distance, useDistances);
comp_time(4) = toc; tic;
[adjmatT{5},edgeD{5}] = RG(distance, useDistances, numSamples);
comp_time(5) = toc;

for i=1:length(adjmatT)
    ll(i) = numSamples*computeLL_Gaussian(sample_cov, edgeD{i});
    bic(i) = ll(i) - (size(adjmatT{i},1)-1)/2 * log(numSamples);
    fprintf('Log-likelihood = %f, BIC score = %f, num_hidden = %d\n',ll(i),bic(i),size(adjmatT{i},1)-m);
    %figure; drawLatentTree(adjmatT{i},m,root,ticker);
end
