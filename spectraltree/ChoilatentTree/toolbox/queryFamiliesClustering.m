function [families, parents, avg_log_ratio] = queryFamiliesClustering(distance,numSamples,verbose)

% Find family groups by adaptive thresholding

if(nargin < 3)
    verbose = 0;
end

edgeD_min = -log(0.1);
edgeD_max = -log(0.9);
m = size(distance,1);
%relD_thres = 2*edgeD_min;  % For reliable statistics, ignore distances below this threshold
relD_thres = -log(0.05)+0.1*log(numSamples);
diff_log_ratio = inf*ones(m);
avg_log_ratio = sparse(m,m);

for i=1:m
    for j=i+1:m
        if(distance(i,j) > 2*edgeD_min)
            diff_log_ratio(i,j) = 10;
            continue;
        end
        if(m > 5)
            other_nodes = (distance(i,:) < relD_thres) & (distance(j,:) < relD_thres);
            dt = relD_thres;
            while(sum(other_nodes) <= 5)  % Need at least 2 other nodes to identify siblings
                dt = dt + log(2);
                other_nodes = (distance(i,:) < dt) & (distance(j,:) < dt);
            end
        else
            other_nodes = true(1,m);
        end
        other_nodes([i,j]) = false;
        log_ratio = distance(i,other_nodes) - distance(j,other_nodes);
        diff_log_ratio(i,j) = max(log_ratio) - min(log_ratio);
        avg_log_ratio(i,j) = mean(log_ratio);  
    end
end

avg_log_ratio = avg_log_ratio - avg_log_ratio';

D = min(diff_log_ratio,diff_log_ratio');
families = kmeansDistance(D,verbose);

% Check whether there exists a parent node for each grouping
parents = zeros(length(families),1);
for f = 1:length(families)
    members = families{f};
    parent_score = zeros(length(members),1);
    for i=1:length(members) 
        p = members(i);
        parent_score(i) = sum(abs(avg_log_ratio(p,members) + distance(p,members)));
    end
    [min_parent_score, j] = min(parent_score);  
    if(length(members)==1 || min_parent_score < 2*edgeD_max*(length(members)-1)) % d(j,h) > edgeD_max
        parents(f) = members(j);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function best_clusters = kmeansDistance(D,verbose)

m = size(D,1);
minD = min(D);
[foo,sort_ind_minD] = sort(minD,'descend');
for i=1:m
    D(i,i) = 0;
end

%max_mean_silh = max(minD)/max(D(:));
max_mean_silh = -log(0.5)/max(D(:));
best_clusters = {1:m};
if(verbose)
    fprintf('k = 1, mean silhouette = %f\n',max_mean_silh);
end
for k = 2:m-2
    for init_ite=1:4
        % Select the initial center points
        if (init_ite==1)
            centers = sort_ind_minD([1:k-1,end])';            
        elseif(init_ite==2)
            centers = sort_ind_minD(1:k)';
        else
            randpermm = randperm(m);
            centers = randpermm(1:k)';
        end
        
        prev_centers = centers;

        for ite=1:5
            % Assign clusters for each point
            clusters = mat2cell(centers,ones(k,1));
            noncenters = setdiff(1:m,centers);
            for j=1:length(noncenters)
                i = noncenters(j);
                [foo, assignC] = min(D(i,centers));
                clusters{assignC}(end+1) = i;
            end

            % Pick a new center for each cluster
            for c=1:k
                minmaxD = inf;
                for j=1:length(clusters{c})
                    i = clusters{c}(j);
                    maxD = max(D(i,clusters{c}));
                    if(maxD < minmaxD)
                        minmaxD = maxD;
                        center = i;
                    end
                end
                centers(c) = center;
            end
            if(isempty(setdiff(centers,prev_centers)))
                break;
            else
                prev_centers = centers;
            end
        end
        mean_silh = compSilhouette(D, clusters);       
 
        if(mean_silh > max_mean_silh)
            max_mean_silh = mean_silh;
            best_clusters = clusters;
            if(verbose)
                fprintf('* ');
                %fprintf('k = %d, mean silhouette = %f\n',k,mean_silh);
                disp(clusters)
            end
        end
        if(verbose)
            fprintf('k = %d, mean silhouette = %f\n',k,mean_silh);
        end
    end    
end
%fprintf('*\n');

function mean_silh = compSilhouette(D, clusters)

m = size(D,1);
k = length(clusters);

sumDcluster = zeros(m,k);
% 
maxinD = zeros(k,1);
for c=1:k
     sumDcluster(:,c) = mean(D(:,clusters{c}),2);
%     %disp(clusters{c})
    maxinD(c) = max(max(D(clusters{c},clusters{c})));
end
maxa = max(maxinD);
silh = zeros(m,1);
for c=1:k
    numMembers = length(clusters{c});
    otherClusterMembers = true(1,m);
    otherClusterMembers(clusters{c}) = false;
    for j=1:numMembers;
        i = clusters{c}(j);
        if(numMembers > 1)
            a = sumDcluster(i,c)*numMembers/(numMembers-1);
            %a = max(D(i,clusters{c}));
        else
            a = maxa;
            %silh(i) = 0;
        end
        b = min(sumDcluster(i,[1:c-1,c+1:end]));
        %b = min(D(i,otherClusterMembers));
        silh(i) = (b-a)/max(a,b);
    end
end
%mean_silh = mean(silh(silh~=0));
mean_silh = mean(silh);