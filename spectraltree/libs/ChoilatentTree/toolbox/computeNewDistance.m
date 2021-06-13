function currDist = computeNewDistance(distance, family_ind, hidden_child_dist_sum)

M = length(hidden_child_dist_sum);
currDist = zeros(M,M);
for i=1:M
    for j=i+1:M
        inter_family_dist_mat = distance(family_ind(i,:),family_ind(j,:));
        inter_family_dist_sum = sum(inter_family_dist_mat(:));
        n1 = size(inter_family_dist_mat,1);
        n2 = size(inter_family_dist_mat,2);
        % Assume that correlation coefficient is bounded above by 0.95
        currDist(i,j) = max((inter_family_dist_sum - hidden_child_dist_sum(i)*n2 - hidden_child_dist_sum(j)*n1)/(n1*n2),-log(0.95));
    end
end

currDist = currDist + currDist';