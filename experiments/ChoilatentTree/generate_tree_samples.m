function X = generate_tree_samples(M,n,psi,type,p)
   

    m = 1+(size(M,1)/2);
    X = zeros(2*m-1,n);
    if type==0 %binary
        X(2*m-1,:) = randsrc(1,n,[1,0 ;p 1-p]);
    else       %discrete
        X(2*m-1,:) = randsrc(1,n,[6:9 ;0.25*ones(1,4)]);
    end
    for i = 1:(m-1)
        parent_idx = m-i;
        child_idx = find(M(:,parent_idx));
        
        if type==0
            pos_idx = find(X(m+parent_idx,:)==1);
            neg_idx = find(X(m+parent_idx,:)==0);        
            X(child_idx(2),pos_idx) = binornd(1,psi(child_idx(2)),1,length(pos_idx));
            X(child_idx(2),neg_idx) = binornd(1,1-psi(child_idx(2)),1,length(neg_idx));
            X(child_idx(1),pos_idx) = binornd(1,psi(child_idx(1)),1,length(pos_idx));
            X(child_idx(1),neg_idx) = binornd(1,1-psi(child_idx(1)),1,length(neg_idx));
        else
            % for each sample, sample positive step, negative step or stay
            % in place
            
            % child 1
            step_1 = randsrc(1,n,[-1 0 1; ...
                (1-psi(child_idx(1)))/2 psi(child_idx(1)) (1-psi(child_idx(1)))/2]); 
            X(child_idx(1),:) = X(m+parent_idx,:)+step_1;
            % child 2
            step_2 = randsrc(1,n,[-1 0 1; ...
                (1-psi(child_idx(2)))/2 psi(child_idx(2)) (1-psi(child_idx(2)))/2]); 
            X(child_idx(2),:) = X(m+parent_idx,:)+step_2;
        end
    end    
end