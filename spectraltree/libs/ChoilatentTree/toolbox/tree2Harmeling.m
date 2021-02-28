function [t_hat] = tree2Harmeling(adjmatT,root,nodePot,edgePot,method,tStructure,tParameter,nodeLabels, trainSamples,testSamples);

tree_msg_order = treeMsgOrder(adjmatT,root);
edge_pairs = tree_msg_order(size(tree_msg_order,1)/2+1:end,:); 
M = size(adjmatT,1);

t_hat.name = method;
t_hat.nobs = size(trainSamples,1);
t_hat.t0 = root;
t_hat.p0 = {nodePot(root,:)'};
t_hat.nsyms = 2*ones(1,M);
children = cell(M,1);
cond_prob = cell(M,1);
for e=1:size(edge_pairs);
    i = edge_pairs(e,1);
    j = edge_pairs(e,2);
    children{i} = [children{i} j];
    k = length(children{i});
    cond_prob{i}{k} = full(edgePot(2*i-1:2*i,2*j-1:2*j))';
end

t_hat.df = 2*M-1;
t_hat.t = children;
t_hat.p = cond_prob;

t_hat.timeStructure = tStructure;
t_hat.timeParameter = tParameter;
t_hat.time = tStructure+tParameter;

dtr.x = trainSamples;
dtr.nsyms = 2*ones(1,t_hat.nobs);

if(nargin == 10)
    dte.x = testSamples;
    dte.nsyms = dtr.nsyms;
    [t_hat.llte, t_hat.missedte] = forrest_ll2(dte, t_hat);
    t_hat.bicte = t_hat.llte - 0.5*t_hat.df*log(size(testSamples,2));
end

[t_hat.lltr, t_hat.missedtr] = forrest_ll2(dtr, t_hat);
t_hat.bictr = t_hat.lltr - 0.5*t_hat.df*log(size(trainSamples,2));


newNodeLabels = cell(M,1);
for i=1:M
    if(i <= t_hat.nobs)
        newNodeLabels{i} = nodeLabels{i};
    else
        newNodeLabels{i} = ['h' num2str(i-t_hat.nobs)];
    end
end

t_hat.nodeLabels = newNodeLabels;
