function [adjmat, level_m] = makeModel(graph, m)

% Generate the adjacency matrix for the given graph with m observed
% variables.

level_m = m;
switch graph
    case 'star'
       M = m+1;
       adjmat = sparse(M,M);
       adjmat(1:m,end) = 1;
    case 'doubleStar'
        M = m+2;
        adjmat = sparse(M,M);
        adjmat(1:ceil(m/2),end-1)=1;
        adjmat(ceil(m/2)+1:end-1,end)=1;
    case 'hmm'
        M = 2*m-2;
        adjmat = sparse(M,M);
        adjmat(1,m+1) = 1;
        adjmat(m,end) = 1;
        adjmat(2:m-1,m+1:M) = speye(m-2);
        adjmat(m+1:M-1,m+2:M) = speye(m-3);
    case 'regular'
        adjmat = sparse(m,m);
        num_nodes = m;

        while(num_nodes > 2)
            num_p = floor(num_nodes/3);
            new_adjmat = sparse(kron(eye(num_p),[1 1 1]));
            res_node = num_nodes - 3*num_p;
            if(res_node == 1)
                new_adjmat = [new_adjmat, [zeros(num_p-1,1); 1]];
            elseif(res_node == 2)
                new_adjmat = [new_adjmat, [zeros(num_p-1,2); 1 1]];
            end
            adjmat = [adjmat; zeros(num_p,size(adjmat,2)-size(new_adjmat,2)), new_adjmat];
            adjmat = [adjmat, zeros(size(adjmat,1),num_p)];
            num_nodes = num_p;
            level_m = [level_m; size(adjmat,1)];
        end
        
        if(num_nodes == 2)
            adjmat(end-1,end) = 1;
        end
        M = size(adjmat,1);
    case '3cayley'
        adjmat = sparse(m,m);
        num_nodes = m;

        while(num_nodes > 2)
            num_p = floor(num_nodes/2);
            new_adjmat = sparse(kron(eye(num_p),[1 1]));
            res_node = num_nodes - 2*num_p;
            if(res_node == 1)
                new_adjmat = [new_adjmat, [zeros(num_p-1,1); 1]];
            end
            adjmat = [adjmat; zeros(num_p,size(adjmat,2)-size(new_adjmat,2)), new_adjmat];
            adjmat = [adjmat, zeros(size(adjmat,1),num_p)];
            num_nodes = num_p;
            level_m = [level_m; size(adjmat,1)];
        end
        
        if(num_nodes == 2)
            adjmat(end-1,end) = 1;
        end
        M = size(adjmat,1);   
    case '5cayley'
        [adjmat,level_m] = makeCayleyTree(m,4);
end

adjmat = adjmat + adjmat';

function [adjmat,level_m] = makeCayleyTree(m,d)

level_m = m;
adjmat = sparse(m,m);
num_nodes = m;

while(num_nodes > 2);
    num_p = floor(num_nodes/d);
    new_adjmat = sparse(kron(eye(num_p),repmat(1,1,d)));
    res_node = num_nodes - d*num_p;
    if(res_node > 0)
        new_adjmat = [new_adjmat, [zeros(num_p-1,res_node); repmat(1,1,res_node)]];
    end
    adjmat = [adjmat; zeros(num_p,size(adjmat,2)-size(new_adjmat,2)), new_adjmat];
    adjmat = [adjmat, zeros(size(adjmat,1),num_p)];
    num_nodes = num_p;
    level_m = [level_m; size(adjmat,1)];
end

if(num_nodes == 2)
    adjmat(end-1,end) = 1;
end