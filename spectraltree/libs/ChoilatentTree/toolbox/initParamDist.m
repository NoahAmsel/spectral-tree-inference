function prob_bij = initParamDist(edgeD, edge_pairs, samples)

% Initialize tree parameters using distances

adjmat = logical(edgeD);
Ntotal = size(adjmat,1);
Nobserved = size(samples,1);
Nsamples = size(samples,2);

prob_bi = zeros(Ntotal,2);
prob_bi(1:Nobserved,2) = sum(samples-1,2)/Nsamples;
for i=Nobserved+1:Ntotal
    neighbors = find(adjmat(i,1:Nobserved));
    if(length(neighbors) > 3)
        votes = sum(samples(neighbors,:)-1,1);
        prob_bi(i,2) = max(sum((votes > length(neighbors)/2))/Nsamples,0.05);
    else
        prob_bi(i,2) = rand(1);
    end
end
prob_bi(:,1) = 1 - prob_bi(:,2);
prob_bij = sparse(2*Ntotal,2*Ntotal);
for e=1:size(edge_pairs,1)
    u = edge_pairs(e,1);
    v = edge_pairs(e,2);  
    prob_bij(2*u-1:2*u,2*v-1:2*v) = findJointProb(edgeD(u,v),prob_bi(u,2),prob_bi(v,2));
end
prob_bi = prob_bi';
prob_bij = prob_bij + prob_bij' + diag(prob_bi(:));


%%%%%%

function jointProb = findJointProb(edge_dist,a,b)

detJoint = exp(-edge_dist + 0.5*sum(log([1-a a 1-b b])));
p11 = detJoint + a*b;
jointProb = [1+p11-a-b, b-p11; a-p11, p11];
if(all(jointProb(:)>=0) && all(jointProb(:) <= 1))
    return;
end
p11 = a*b - detJoint;
jointProb = [1+p11-a-b, b-p11; a-p11, p11];
if(all(jointProb(:)>=0) && all(jointProb(:) <= 1))
    return;
end    

minP = max(0,a+b-1);
maxP = min(a,b);
p11 = (maxP-minP)*rand(1)+minP;
jointProb = [1+p11-a-b, b-p11; a-p11, p11];
