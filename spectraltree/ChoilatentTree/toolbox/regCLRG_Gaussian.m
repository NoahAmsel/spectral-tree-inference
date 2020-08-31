function [adjmatTree, edge_distance] = regCLRG_Gaussian(distance, numSamples, verbose, maxHidden)

if(nargin < 3)
    verbose = 0;
end

if(nargin < 4)
    maxHidden = inf;
end


cov_mat = exp(-distance);
m = size(distance,1);
adjmatCL = ChowLiu(-distance);
edge_distance = distance.*adjmatCL;
ll = computeLL_Gaussian(cov_mat, edge_distance);
bic = numSamples*ll - (m-1)/2 * log(numSamples);

degree = sum(adjmatCL,2);
internal_nodes = find(degree > 1);
num_nodes = size(adjmatCL,1);
surrogate_nodes = [];
dist2surrogate = [];

while(~isempty(internal_nodes) && num_nodes <= m+maxHidden)
    edge_distance_cand = cell(length(internal_nodes),1);
    bic_cand = -inf * ones(length(internal_nodes),1);
    for j=1:length(internal_nodes)
    
        % Select neighbors of an internal node
        i = internal_nodes(j);
        i_family = union(i,find(edge_distance(i,:)));
        
        % Replace hidden neighbors with their surrogate nodes
        isHidden = (i_family > m);
        hidden_nodes = i_family(isHidden);
        i_family(isHidden) = surrogate_nodes(hidden_nodes-m);
    
        % Apply recursive grouping to i and its neighbors
        [adjmatS, edge_distanceS] = RG(distance(i_family,i_family),1, numSamples);
        num_new_nodes = size(adjmatS,1) - length(i_family);
        edge_distance_cand{j} = [edge_distance, sparse(size(edge_distance,1),num_new_nodes)];
        edge_distance_cand{j} = [edge_distance_cand{j}; sparse(num_new_nodes,size(edge_distance_cand{j},2))];
    
        % Replace surrogate nodes back to their hidden nodes
        i_family(isHidden) = hidden_nodes;
        hidden_index = find(isHidden);
        for h=1:length(hidden_index)
            h_ind = hidden_index(h);
            h_neigh = logical(edge_distanceS(h_ind,:));
            h_node = hidden_nodes(h);
            new_ed = max(edge_distanceS(h_ind,h_neigh) - dist2surrogate(h_node-m),-log(0.95));
            edge_distanceS(h_ind,h_neigh) = new_ed;
            edge_distanceS(h_neigh,h_ind) = new_ed;
        end

        % Update the edge distance matrix
        new_node_ind = num_nodes+1:num_nodes+num_new_nodes;    
        i_new_family = [i_family, new_node_ind];
        edge_distance_cand{j}(i_new_family,i_new_family) = edge_distanceS;  
        ll = computeLL_Gaussian(cov_mat, edge_distance_cand{j});
        bic_cand(j) = numSamples*ll - (num_nodes+num_new_nodes-1)/2 * log(numSamples);
    end
    [max_bic,ind] = max(bic_cand);
    if(max_bic < bic(end))
        if(verbose)
            fprintf('BIC not increasing. Terminating the iterations...\n');
            fprintf('Best candidate node %d, BIC = %f\n',internal_nodes(ind),max_bic);
        end
        break;
    end
    i = internal_nodes(ind);
    if(verbose)
        fprintf('NJ to node %d, BIC = %f\n',i,max_bic);
    end
    internal_nodes(ind) = [];
    bic(end+1) = max_bic;
    num_new_nodes = size(edge_distance_cand{ind},1) - size(edge_distance,1);
    edge_distance = edge_distance_cand{ind};

    new_node_ind = num_nodes+1:num_nodes+num_new_nodes; 
    surhid = [i, new_node_ind];
    subtree_dist = treeDistance(edge_distance(surhid,surhid));
    dist2surrogate = [dist2surrogate; subtree_dist(2:end,1)];
    surrogate_nodes = [surrogate_nodes; i*ones(num_new_nodes,1)];      
    num_nodes = num_nodes+num_new_nodes;
end

edge_distance = contractWeakEdges(edge_distance,m);

adjmatTree = logical(edge_distance);

