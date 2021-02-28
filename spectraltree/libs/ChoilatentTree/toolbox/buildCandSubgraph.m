function [candGraph, ll_new] = buildCandSubgraph(onodes, edge_pairs, samples, edge_distanceS)

% Learn parameters over a latent tree
% nodes = index of observed nodes in the big graph
% edge_pairs = orders of edges in the big graph
% samples = samples of onodes
% edge_distanceS = edge_distance matrix over the observed and hidden nodes

%num_nodes = size(edge_pairs,1)+1;
num_new_nodes = size(edge_distanceS,1) - length(onodes);      
%new_node_ind = num_nodes+1:num_nodes+num_new_nodes;    
%i_new_family = [onodes, new_node_ind];         

[belongsNodes, ind] = ismember(edge_pairs(:,1), onodes);
edgeNum = find(belongsNodes,1);
subroot_ind = ind(edgeNum);  

candGraph.subrootInd = subroot_ind;
candGraph.nodeInd = onodes;
candGraph.numNewNodes = num_new_nodes;
candGraph.edgeDistance = edge_distanceS;
 
options.max_ite = max(10*num_new_nodes,5);
options.root = subroot_ind;
if(~islogical(edge_distanceS(1,1)))
    options.edge_distance = edge_distanceS;
end
[candGraph.nodePot, candGraph.edgePot, ll_new] = learnParamsEMmex(samples,logical(edge_distanceS),options);

