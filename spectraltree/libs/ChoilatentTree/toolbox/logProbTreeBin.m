function logP = logProbTreeBin(root_mar,edgePot,edge_pairs,samples)

numSamples = size(samples,2);
logP = zeros(numSamples,1);

root = edge_pairs(1,1);
tiny = 1e-10;
for n=1:numSamples
    logP(n) = logP(n) + log(root_mar(samples(root,n))+tiny);
    for e=1:size(edge_pairs,1)
        p = edge_pairs(e,1);
        c = edge_pairs(e,2);
        logP(n) = logP(n) + log(edgePot(2*p-2+samples(p,n),2*c-2+samples(c,n))+tiny);
    end
end