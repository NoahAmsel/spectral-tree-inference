function [edge_distance, edge_distance_sum] = computeHiddenChild(child_pair_distance, child_pair_diff)

num_childs = size(child_pair_distance,1);
if(num_childs == 2)
    edge_distance_sum = child_pair_distance(1,2);
    edge_distance = 0.5*[1 1; 1 -1]*[edge_distance_sum; child_pair_diff]; 
else
    edge_distance = sum(child_pair_distance,2);
    edge_distance_sum = sum(edge_distance)/(2*(num_childs-1));
    edge_distance = (edge_distance - edge_distance_sum)/(num_childs-2);
end

% Assume that correlation coefficient is bounded above by 0.95
edge_distance = max(edge_distance, -log(0.95)); 
