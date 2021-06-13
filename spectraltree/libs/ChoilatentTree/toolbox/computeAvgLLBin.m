function ll = computeAvgLLBin(rootnode_potential,edge_potential,prob_bij,edge_pairs)

% Computes the average log-likelihood of a binary tree.
% In order to get log-likelihood, need to multiply by the number of
% samples.

tiny = 10e-20;
root = edge_pairs(1,1);
log_prob = log(rootnode_potential(:)+tiny);
empirical_prob = diag(prob_bij(2*root-1:2*root,2*root-1:2*root));
ll = log_prob(:)'*empirical_prob(:);

for e=1:size(edge_pairs,1)
    i = edge_pairs(e,1);
    j = edge_pairs(e,2);

    log_cond_prob = log(edge_potential(2*i-1:2*i,2*j-1:2*j)+tiny);
    empirical_prob = prob_bij(2*i-1:2*i,2*j-1:2*j);
    ll = ll + log_cond_prob(:)'*empirical_prob(:);
end


