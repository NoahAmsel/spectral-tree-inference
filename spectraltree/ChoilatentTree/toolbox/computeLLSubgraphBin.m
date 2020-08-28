function ll = computeLLSubgraphBin(nodeSubset, samples, edge_pairs, edgePot)

% Compute the log-likelihood of a subgraph given by nodeSubset 


[inSubset1,ind1] = ismember(edge_pairs(:,1),nodeSubset);
[inSubset2,ind2] = ismember(edge_pairs(:,2),nodeSubset);
inSubset = inSubset1 & inSubset2;
edge_pairsSG = [ind1(inSubset), ind2(inSubset)];

nodeSG_states = [2*nodeSubset-1;2*nodeSubset];
s = nodeSG_states(:);
subroot = edge_pairsSG(1,1);
subroot_mar = sum(samples(subroot,:)-1)/size(samples,2);
subroot_mar = [1-subroot_mar, subroot_mar];

ll = sum(logProbTreeBin(subroot_mar,edgePot(s,s),edge_pairsSG,samples));
