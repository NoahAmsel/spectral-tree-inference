function tree_distance = treeDistance(adjmat, edge_distance)

% Compute distance between all pairs of variables given the
% adjacency matrix of a tree and parent-child distances

if(nargin < 2)
    edge_distance = adjmat;
end

adjmat = logical(adjmat);
num_nodes = size(adjmat,1);

if(num_nodes == 1)
    tree_distance = 0;
    return
elseif(num_nodes == 2)
    tree_distance = [0 edge_distance(1,2); edge_distance(2,1) 0];
    return
end
tree_distance = zeros(num_nodes,num_nodes);
degree = sum(adjmat,2);
leaf_nodes = find(degree ==1);
if(isempty(leaf_nodes))
    degree
    error('No leaf node\n');
end
l = leaf_nodes(1);
other_nodes = true(num_nodes,1);
other_nodes(l) = false;

sub_distance = treeDistance(adjmat(other_nodes,other_nodes), edge_distance(other_nodes,other_nodes));
tree_distance(other_nodes,other_nodes) = sub_distance;
p = adjmat(l,:);
tree_distance(l,:) = tree_distance(p,:)+edge_distance(l,p);
tree_distance(l,l) = 0;
tree_distance(:,l) = tree_distance(l,:)';
