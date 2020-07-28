function X = generate_tree_samples_multiclass(M,n,T,p)
   
    m = 1+(size(M,1)/2);
    X = zeros(2*m-1,n);
    K = size(T,1);
    
    % sample root data
    %X(2*m-1,:) = randsrc(1,n,[1:K ; p]);
    X(2*m-1,:) = randi(K,1,n);
    
    % sample evolutionary data
    for i = 1:(m-1)
        parent_idx = m-i;
        child_idx = find(M(:,parent_idx));
        
        % scan over all possible values and fill values for both
        % descendents
        for k = 1:K
            k_idx = find(X(m+parent_idx,:)==k);
            X(child_idx(2),k_idx) = randsrc(1,length(k_idx),[1:K ; T(:,k,child_idx(2))']);            
            X(child_idx(1),k_idx) = randsrc(1,length(k_idx),[1:K ; T(:,k,child_idx(1))']);
        end        
        
    end    
end