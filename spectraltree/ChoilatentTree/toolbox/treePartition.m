function partitionSet = treePartition(adjmat,root,m)

% Computes the partitions of observed ndoes 1:m for each edge to compute
% the Robinson-Foulds metric

if(root==0)
    degree = sum(adjmat,1);
    [foo,root] = max(degree);
end

children = find(adjmat(root,:));
if(isempty(children))
    partitionSet = false(1,m);
    partitionSet(root) = true;
    return
end
adjmat(root,children) = 0;
adjmat(children,root) = 0;
partitionSet = [];
partitionSet_root = false(1,m);
if(root <= m)
    partitionSet_root(root) = true;
end
for c=1:length(children)
    partitionSet_c = treePartition(adjmat,children(c),m);
    partitionSet = [partitionSet; partitionSet_c];
    partitionSet_root = partitionSet_root | partitionSet_c(1,:);
end

partitionSet = [partitionSet_root; partitionSet];
    

%  dist = sum(ismember(treePartition1, treePartition2, 'rows'));