function newSamples = sampleFromHidden(nodePot, edgePot, samples, adjmat, root)

Ntotal = size(adjmat,1);
Nobserved = size(samples,1);
Nsamples = size(samples,2);
hnodes = Nobserved+1:Ntotal;
newSamples = zeros(Ntotal,Nsamples);
tree_msg_order = treeMsgOrder(adjmat,root);

for n=1:Nsamples
    node_pot = nodePot;
    node_pot(1:Nobserved,:) = repmat([1 0],Nobserved,1);    
    indSample2 = (samples(:,n)==2);
    node_pot(indSample2,1) = 0;
    node_pot(indSample2,2) = 1;

    newSamples(:,n) = sampleFromBinTree(adjmat,node_pot,edgePot,tree_msg_order);

end
newSamples = newSamples(hnodes,:);