function [adjmatTree] = NJb(stats, useDistances)

% Recursive grouping algorithm to learn latent trees
% PARAMETERS:
%       if useDistances==true:  
%           stats = information distance matrix of observed nodes
%       if useDistances==false:
%           stats = samples of binary observed variables
%
% OUTPUTS:
%       adjmatTree = an adjacency matrix of a tree including latent nodes
%       edge_distance = information distances on the edges of the tree
%
% Myung Jin Choi, Jan 2010, MIT

if useDistances
    distance = stats;
else
    samples = stats;
    prob_bij = computeBnStats(samples);
    distance = computeDistance(prob_bij);    
end

verbose = 0;
edgeD_max = -log(0.9);
m = size(distance,1); % # observed nodes

edge_distance = sparse(m,m);
D = distance;

numNodes = m;
newNodeNum = m+1;
nodeNum = 1:m;
rd = sum(D,1);
while(numNodes>2)
    rdmat = repmat(rd,numNodes,1);
    Q = D - (rdmat+rdmat')/(numNodes-2); 
    Q = Q - diag(diag(Q));
    [minQ, minPairInd] = min(Q(:));
    [ind(1),ind(2)] = ind2sub(size(Q),minPairInd);
    i = nodeNum(ind(1));
    j = nodeNum(ind(2));
    %disp(i); disp(j);

    edist = 0.5*(D(ind(1),ind(2))*[1, 1] + (rd(ind(1))-rd(ind(2)))/(numNodes-2)*[1 -1]);
    new_distance = 0.5*(D(ind(1),:)+D(ind(2),:)) - 0.5*D(ind(1),ind(2));
    new_distance(ind) = [];
    
    if(edist > edgeD_max)     
        D(ind,:) = [];
        D(:,ind) = [];     
        nodeNum(ind) = [];           
        if(new_distance > edgeD_max)
            p_node = newNodeNum;      
            newNodeNum = newNodeNum+1;
            nodeNum(end+1) = p_node;
            edge_distance = [edge_distance, sparse(size(edge_distance,1),1)];
            edge_distance = [edge_distance; sparse(1,size(edge_distance,2))];             
            edge_distance(p_node,[i,j]) = edist;
            if(verbose)
                fprintf('Merging %d and %d, new parent node %d\n',i,j,p_node);
            end
            D = [D, new_distance'; new_distance, 0];
        else
           [foo, indp] = min(new_distance);
           p_node = nodeNum(indp);
           edge_distance(p_node,[i,j]) = edist;
           numNodes = numNodes-1;
           if(verbose)
               fprintf('Merging %d and %d, under the parent node %d\n',i,j,p_node);        
           end
        end
    else
        [edge_distance(j,i),p] = max(edist);
        D(ind(p),:) = [];
        D(:,ind(p)) = [];
        if(verbose)
            fprintf('%d is a parent of %d\n',nodeNum(ind(3-p)),nodeNum(ind(p)));
        end
        nodeNum(ind(p)) = [];
    end
    rd = sum(D,1);
    numNodes = numNodes - 1;
end

if(length(nodeNum)==2)
    edge_distance(nodeNum(1),nodeNum(2)) = D(1,2);
    if(verbose)
        fprintf('Connecting %d and %d\n',nodeNum(1),nodeNum(2));
    end
end
    
edge_distance = edge_distance + edge_distance';
edge_distance = edge_distance - diag(diag(edge_distance));
#adjmatTree = logical(edge_distance);
adjmatTree = edge_distance;

