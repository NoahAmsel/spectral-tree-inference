function [adjmatTree, edge_distance, nodePot, edgePot, ll, bic] = regCLRG_discrete(samples, options)

numSamples = size(samples,2);
m = size(samples,1);

if ~isfield(options, 'root')
    root = 1;
else
    root = options.root;
end

if ~isfield(options, 'verbose')
    options.verbose = 0;
end

if ~isfield(options, 'maxHidden')
    options.maxHidden = inf;
end

if ~isfield(options, 'nodeLabels')
    options.nodeLabels = cellstr([int2str((1:m)') repmat(['   '],m,1)]);
end

if ~isfield(options, 'startingTree')
    options.startingTree = 'MST';
end

prob_bij = computeBnStats(samples);
distance = computeDistance(prob_bij);
if (strcmp(options.startingTree,'MST'))
    adjmatCL = ChowLiu(-distance);
elseif(strcmp(options.startingTree, 'CL'))
    mi = computeMutualInformationBin(prob_bij);
    adjmatCL = ChowLiu(mi);
end

num_nodes = size(adjmatCL,1);
edge_distance = distance.*adjmatCL;
tree_msg_order = treeMsgOrder(adjmatCL,root);
edge_pairs = tree_msg_order(num_nodes:end,:);
[nodePot, edgePot] = marToPotBin(prob_bij,tree_msg_order);

ll = numSamples*computeAvgLLBin(nodePot(root,:),edgePot,prob_bij,edge_pairs);
bic = ll - 0.5*(2*num_nodes-1)*log(numSamples); 

samplesH = samples;
adjmatTree = adjmatCL;
degree = sum(adjmatTree,2);
internal_nodes = find(degree > 1);
surrogate_nodes = [];
dist2surrogate = [];

candGraph = cell(num_nodes,1);
bic_inc = -inf*ones(num_nodes,1);
ll_diff = -inf*ones(num_nodes,1);
list_cand_nodes = internal_nodes;

while(num_nodes <= m+options.maxHidden)
    
    for j=1:length(list_cand_nodes)
    
        % Select neighbors of an internal node
        i = list_cand_nodes(j);
        i_family = union(i,find(adjmatTree(i,:)));
        
        % Replace hidden neighbors with their surrogate nodes
        isHidden = (i_family > m);
        hidden_nodes = i_family(isHidden);
        i_family(isHidden) = surrogate_nodes(hidden_nodes-m);
    
        % Apply recursive grouping to i and its neighbors
        [adjmatS, edge_distanceS] = RG(distance(i_family,i_family),1, numSamples);
    
        % Replace surrogate nodes back to their hidden nodes
        i_family(isHidden) = hidden_nodes;
        hidden_index = find(isHidden);
        for h=1:length(hidden_index)
            h_ind = hidden_index(h);
            h_neigh = adjmatS(h_ind,:);
            h_node = hidden_nodes(h);
            new_ed = max(edge_distanceS(h_ind,h_neigh) - dist2surrogate(h_node-m),-log(0.95));
            edge_distanceS(h_ind,h_neigh) = new_ed;
            edge_distanceS(h_neigh,h_ind) = new_ed;
        end       
        
        % Construct a candidate graph and compute the difference in BIC
        [candGraph{i}, ll_new] = buildCandSubgraph(i_family, edge_pairs, samplesH(i_family,:), edge_distanceS);
        ll_old = computeLLSubgraphBin(i_family, samplesH(i_family,:), edge_pairs, edgePot);
        
        ll_diff(i) = ll_new(end) - ll_old;
        num_new_nodes = candGraph{i}.numNewNodes;
        if(num_new_nodes > 0)
            bic_inc(i) = ll_diff(i) - 0.5*(2*num_new_nodes-1)*log(numSamples);
        else
            ll_diff(i) = 0;
            bic_inc(i) = ll_diff(i);
        end
        fprintf('%s, %.2f, %.2f\n',options.nodeLabels{i},ll_diff(i),bic_inc(i));
    end
    
    [max_bic_inc,ind] = max(bic_inc);
    if(max_bic_inc <= 0)
        if(options.verbose)
            fprintf('BIC not increasing. Terminating the iterations...\n');
            fprintf('Best candidate node %d, BIC = %f\n',ind,max_bic_inc);
        end
        break;
    end
    if(options.verbose)
        fprintf('Selected NJ to node %s, BIC increment = %f\n',options.nodeLabels{ind},max_bic_inc);
    end
    list_cand_nodes = intersect(setdiff(candGraph{ind}.nodeInd,ind),internal_nodes);
    bic(end+1) = max_bic_inc+bic(end);
    bic_inc(ind) = -inf;
    ll(end+1) = ll(end)+ll_diff(ind);
    
    % Update nodePot, edgePot, samplesH, edge_distance, edge_pairs
    num_new_nodes = candGraph{ind}.numNewNodes;
    new_node_ind = num_nodes+1:num_nodes+num_new_nodes; 
    num_nodes = num_nodes+num_new_nodes;
    nodeSubset = [candGraph{ind}.nodeInd, new_node_ind];
    edge_distance(nodeSubset,nodeSubset) = candGraph{ind}.edgeDistance;  
    adjmatTree = logical(edge_distance);

    nodePot(nodeSubset,:) = candGraph{ind}.nodePot;
    nodeSG_states = [2*nodeSubset-1;2*nodeSubset];
    s = nodeSG_states(:);
    edgePot(s,s) = candGraph{ind}.edgePot;
    newSamples = sampleFromHidden(candGraph{ind}.nodePot, candGraph{ind}.edgePot, samplesH(candGraph{ind}.nodeInd,:), logical(candGraph{ind}.edgeDistance), candGraph{ind}.subrootInd);
    samplesH = [samplesH; newSamples];    
    tree_msg_order = treeMsgOrder(adjmatTree,root);
    edge_pairs = tree_msg_order(num_nodes:end,:); 

    surhid = [ind, new_node_ind];
    subtree_dist = treeDistance(edge_distance(surhid,surhid));
    dist2surrogate = [dist2surrogate; subtree_dist(2:end,1)];
    surrogate_nodes = [surrogate_nodes; ind*ones(num_new_nodes,1)];      
 
end