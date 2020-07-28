function [is_exact, path_error_sum] = isExactRecovery(adjmatTree, topo_distance_org)

m = size(topo_distance_org,1);
topo_distance_est = treeDistance(adjmatTree);
path_error = abs(topo_distance_est(1:m,1:m) - topo_distance_org);
%diameter_org = max(topo_distance_org(:));
%diameter_est = max(topo_distance_est(:));
path_error_sum = sum(path_error(:))/(numel(path_error));
if(all(~logical(path_error(:))))
    is_exact = true;
else
    is_exact = false;
end