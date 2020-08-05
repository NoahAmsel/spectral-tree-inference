function adjmat = ChowLiu(mi)

% Learn a tree using Chow-liu algorithm

N = size(mi,1);

mi = mi - 1000*eye(N);
[mi_sorted, index] = sort(mi(:), 'descend');

N = size(mi,1);
adjmat = zeros(N);
num_edge=0;
i=1;
while(num_edge < N-1)
    [sub1,sub2] = ind2sub([N,N],index(i));
    if(~connected(adjmat,sub1,sub2))
        adjmat(sub1,sub2) = 1;
        adjmat(sub2,sub1) = 1;
        num_edge = num_edge+1;
    end
    i = i+1;
end