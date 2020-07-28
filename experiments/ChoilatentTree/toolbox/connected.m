function tf = connected(adj,i,j)

% Returns true of i and j are connected on the graph.

adj = logical(adj);
if(adj(i,j))
    tf=true;
    return
end
neighbors = adj(i,:);
adj(i,:) = false;
adj(:,i) = false;
tf = false;

while(any(neighbors))
    if(neighbors(j))
        tf = true;
        break;
    end
    new_neighbors = any(adj(neighbors,:),1);
    adj(neighbors,:) = false;
    adj(:,neighbors) = false;
    neighbors = new_neighbors;
end
