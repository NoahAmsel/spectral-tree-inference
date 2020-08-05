function ll = computeLL_Gaussian(sample_cov, edge_distance)

m = size(sample_cov,1);

pairD = treeDistance(edge_distance);
J = inv(exp(-pairD(1:m,1:m)));
ll = -0.5*trace(J*sample_cov) + 0.5*log(det(J)) - m/2 * log(2*pi);