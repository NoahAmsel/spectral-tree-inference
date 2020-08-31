function drawLatentTree(adjmat,m, root_index, nodeLabels)

num_nodes = size(adjmat,1);
if(nargin < 2)
    m = num_nodes;
end
if(nargin < 3)
    root_index = num_nodes;
end
if(nargin < 4)
    nodeLabels = cellstr([int2str((1:num_nodes)') repmat(['   '],num_nodes,1)]);
end
if(length(nodeLabels) < size(adjmat,1))
    num_hidden_nodes = size(adjmat,1) - length(nodeLabels);
    hiddenNodeLabels = cellstr([int2str((1:num_hidden_nodes)') repmat(['   '],num_hidden_nodes,1)]);
    nodeLabels = [nodeLabels; hiddenNodeLabels];
end

edge_weight = 0.1*adjmat;
node_box = zeros(num_nodes,1);
node_box(m+1:end) = 1;

[x,y,h] = drawWeightedGraph(adjmat, nodeLabels, root_index, edge_weight, node_box);

