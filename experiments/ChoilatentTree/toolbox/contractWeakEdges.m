function edge_distance = contractWeakEdges(edge_distance,m)

edgeD_max = -log(0.9);
[ind1,ind2,s] = find(edge_distance);
weak_ind = find(s < edgeD_max & ind1 < ind2);

nodes1 = ind1(weak_ind);
nodes2 = ind2(weak_ind);
while(~isempty(nodes1))
    if(nodes1(end) > m) % node1(k) is a hidden node, so remove node1(k)
        i = nodes1(end);
        j = nodes2(end);
    elseif(nodes2(end) > m);
        i = nodes2(end);
        j = nodes1(end);
    else
        nodes1(end) = [];
        nodes2(end) = [];        
        continue;
    end
    edge_distance(j,:) = edge_distance(j,:) + edge_distance(i,:);
    edge_distance(:,j) = edge_distance(j,:)';
    edge_distance(j,j) = 0;
    edge_distance(i,:) = [];
    edge_distance(:,i) = [];
    %fprintf('Merging %d to %d\n',i,j);
    
    nodes1(end) = [];
    nodes2(end) = [];
    nodes1(nodes1==i) = j;
    nodes2(nodes2==i) = j;
    nodes1(nodes1 > i) = nodes1(nodes1 > i) -1;
    nodes2(nodes2 > i) = nodes2(nodes2 > i) -1;
end

% 
% rnode = []; mnode = [];
% for k=1:length(weak_ind)
%     node1 = ind1(weak_ind(k));
%     node2 = ind2(weak_ind(k));
%     if(node1 > m) % node1 is a hidden node
%         rnode(end+1) = node1;
%         mnode(end+1) = node2;
%     elseif(node2 > m)
%         rnode(end+1) = node2;
%         mnode(end+1) = node1;
%     else
%         continue;        
%     end    
% end
% 
% edge_distance(mnode,:) = edge_distance(rnode,:) + edge_distance(mnode,:);
% edge_distance(:,mnode) = edge_distance(mnode,:)';
% for k=1:length(mnode)
%     edge_distance(mnode(k),mnode(k)) = 0;
% end
% edge_distance(rnode,:) = [];
% edge_distance(:,rnode) = [];

