function distance = computeDistance(jointProb,marProb)

if(nargin < 2)
    marProb = diag(jointProb);
    marProb = reshape(marProb,2,numel(marProb)/2)';
end

N = size(marProb,1);
K = size(marProb,2);

distance = zeros(N);
tiny = 1e-10;
for i=1:N
    for j=i+1:N
        joint_prob = jointProb(K*(i-1)+1:K*i,K*(j-1)+1:K*j);
        distance(i,j) = -log(abs(det(joint_prob))/sqrt(prod(marProb(i,:))*prod(marProb(j,:))+tiny)+tiny);
    end
end

distance = distance + distance';