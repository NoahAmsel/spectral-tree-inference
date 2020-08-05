function [adjmat,newNodeLabels] = harmelingTree2Adjmat(t, nodeLabels)

roots = t.t0;                  % the list of roots
children = t.t;                % the list of children for each node
nsyms = t.nsyms;
nforests = length(roots);

n = length(children);          % number of nodes
newNodeLabels = cell(n,1);
adjmat = sparse(n,n);
for i=1:n
    adjmat(i,children{i}) = 1;
    if(i <= t.nobs)
        newNodeLabels{i} = nodeLabels{i};
    else
        newNodeLabels{i} = ['h' num2str(i-t.nobs)];
    end
    newNodeLabels{i} = [newNodeLabels{i} '(' num2str(nsyms(i)) ')'];
end
adjmat = adjmat + adjmat';

figure;
for r=1:nforests  
    r12 = mod(r,12);
    if(r12==0)
        figure
    end
    subplot(3,4,r12+1);
    stack = roots(r);
    prev_nodes = stack;
    added_nodes = stack;
    while(~isempty(added_nodes))
        added_nodes = [];
        for i=1:length(prev_nodes)
            added_nodes = [added_nodes, children{prev_nodes(i)}];
        end
        stack = [stack, added_nodes];
        prev_nodes = added_nodes;
    end
    if(length(stack)==1)
        adjmat(stack,stack) = 1;
    end
    drawLatentTree(adjmat(stack,stack),length(stack<t.nobs),1,newNodeLabels(stack));
end
