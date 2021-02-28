function T = generate_transition_matrices(K,m,pr_bounds,type)
    
    T = zeros(K,K,2*m-2);
    if type==0 % totally random, diagonal elements chosen from a
        % pre - defined range
        
        for i = 1:(2*m-2)
            for j = 1:K
                T(j,j,i) = pr_bounds(1) + diff(pr_bounds)*rand();
                p_i = rand(K-1,1);
                T([1:(j-1) (j+1):K],j,i) = (1-T(j,j,i))*p_i/sum(p_i);
            end
        end
        
    elseif type==1 %single value for elements in diagonal,
        % single value for off diagonal
        for i = 1:(2*m-2)
            p = pr_bounds(1) + diff(pr_bounds)*rand();
            T_i = zeros(K);
            T_i(logical(eye(K)))=p;
            T_i(logical(1-eye(K))) = (1-p)/(K-1);
            T(:,:,i) = T_i;                  
        end        
    end
    
end