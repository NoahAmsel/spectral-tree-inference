function atree = create_tree(treenumber);
% generate data from a tree structured distribution

% use a matrix to form the tree
% for each node in the tree there is a row, which lists all children
% leaves have no row and must have higher numbers than inner nodes
% we also need to fix the number of symbols a node can have
% all trees must be complete, i.e. an empty LHS or RHS is not allowed
% right now only trees with equal number of kids is possible
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

switch treenumber
 case 1  % binary-forest
  %          7          8
  %         / \        / \
  %        1   6      4   5
  %           / \
  %          2   3
  t0 = [7, 8];   % the root
  t = cell(8, 1);
  t{8} = [4 5];
  t{7} = [1 6];
  t{6} = [2 3];
  nsyms = [4 8 8 4 4 4 2 2];
  nobs = 5;
  p0 = cell(1, 2);
  p0{1} = [.5; .5];
  p0{2} = [.5; .5];
  p = cell(8, 1);
  p{8} = cell(1, 2);
  p{8}{1} = [.5, .0; 
             .5, .0;
             .0, .5;
             .0, .5];
  p{8}{2} = p{8}{1};
  p{7}{1} = [.5, .0; 
             .5, .0;
             .0, .5;
             .0, .5];
  p{7}{2} = p{7}{1};
  p{6} = cell(1, 2);
  p{6}{1} = [.5, .0, .0, .0; 
             .5, .0, .0, .0;
             .0, .5, .0, .0;
             .0, .5, .0, .0;
             .0, .0, .5, .0;
             .0, .0, .5, .0;
             .0, .0, .0, .5;
             .0, .0, .0, .5];
  p{6}{2} = p{6}{1};
 
 case 2  % three-level-binary
  %            7        
  %          /   \      
  %        5       6    
  %       / \     / \   
  %      1   2   3   4  
  t0 = [7];   % the root
  t = cell(7, 1);
  t{7} = [5 6];
  t{6} = [3 4];
  t{5} = [1 2];
  nsyms = [8, 8, 8, 8, 4, 4, 2];
  nobs = 4;
  p0 = cell(1, 1);
  p0{1} = [.5; .5];
  p = cell(7, 1);
  p{7} = cell(1, 2);
  p{7}{1} = [.5, .0; 
             .5, .0;
             .0, .5;
             .0, .5];
  p{7}{2} = p{7}{1};
  p{6} = cell(1, 2);
  p{6}{1} = [.5, .0, .0, .0; 
             .5, .0, .0, .0;
             .0, .5, .0, .0;
             .0, .5, .0, .0;
             .0, .0, .5, .0;
             .0, .0, .5, .0;
             .0, .0, .0, .5;
             .0, .0, .0, .5];
  p{6}{2} = p{6}{1};
  p{5} = cell(1, 2);
  p{5}{1} = p{6}{1};
  p{5}{2} = p{5}{1};
  
 case 3   % three-coins
  %
  %          4     
  %        / | \   
  %       1  2  3  
  t0 = [4];
  t = cell(4, 1);
  t{4} = [1 2 3];
  nsyms = [4 4 4 8];
  nobs = 3;
  p0 = cell(1, 1);
  p0{1} = ones(8, 1)/8;
  p = cell(4, 1);
  p{4} = cell(1, 3);
  p{4}{1} = [1 0 0 0 1 0 0 0;
             0 1 0 0 0 1 0 0;
             0 0 1 0 0 0 1 0;
             0 0 0 1 0 0 0 1];
  p{4}{2} = [1 0 1 0 0 0 0 0;
             0 1 0 1 0 0 0 0;
             0 0 0 0 1 0 1 0;
             0 0 0 0 0 1 0 1];
  p{4}{3} = [1 1 0 0 0 0 0 0;
             0 0 1 1 0 0 0 0;
             0 0 0 0 1 1 0 0;
             0 0 0 0 0 0 1 1];             
  
 case 4   % six-coins
  %
  %          6       
  %        / | \     
  %       1  2  5    
  %            / \   
  %           3   4  
  t0 = [6];
  t = cell(6, 1);
  t{6} = [1 2 5];
  t{5} = [3 4];
  nsyms = [8 8 8 8 16 16];
  nobs = 4;
  p0 = cell(1, 1);
  p0{1} = ones(16, 1)/16;
  p = cell(6, 1);
  p{6} = cell(1, 3);
  % node 1 ignores the 4th bit of node 6
  p{6}{1} = [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1];
  % node 2 ignores the 3th bit of node 6
  p{6}{2} = [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
             0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0;
             0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1];
  % node 5 ignores the 1st and 2nd bit of node 6
  p{6}{3} = [.25 .25 .25 .25  0   0   0   0   0   0   0   0   0   0   0   0 ;
             .25 .25 .25 .25  0   0   0   0   0   0   0   0   0   0   0   0 ;
             .25 .25 .25 .25  0   0   0   0   0   0   0   0   0   0   0   0 ;
             .25 .25 .25 .25  0   0   0   0   0   0   0   0   0   0   0   0 ;
              0   0   0   0  .25 .25 .25 .25  0   0   0   0   0   0   0   0 ;
              0   0   0   0  .25 .25 .25 .25  0   0   0   0   0   0   0   0 ;
              0   0   0   0  .25 .25 .25 .25  0   0   0   0   0   0   0   0 ;
              0   0   0   0  .25 .25 .25 .25  0   0   0   0   0   0   0   0 ;
              0   0   0   0   0   0   0   0  .25 .25 .25 .25  0   0   0   0 ;
              0   0   0   0   0   0   0   0  .25 .25 .25 .25  0   0   0   0 ;
              0   0   0   0   0   0   0   0  .25 .25 .25 .25  0   0   0   0 ;
              0   0   0   0   0   0   0   0  .25 .25 .25 .25  0   0   0   0 ;
              0   0   0   0   0   0   0   0   0   0   0   0  .25 .25 .25 .25;
              0   0   0   0   0   0   0   0   0   0   0   0  .25 .25 .25 .25;
              0   0   0   0   0   0   0   0   0   0   0   0  .25 .25 .25 .25;
              0   0   0   0   0   0   0   0   0   0   0   0  .25 .25 .25 .25];
  % node 3 ignores the 4th bit of node 5
  p{5}{1} = [1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0;
             0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0;
             0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0;
             0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0;
             0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0;
             0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0;
             0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0;
             0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1];
  % node 4 ignores the 3th bit of node 5
  p{5}{2} = [1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0;
             0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0;
             0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0;
             0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0;
             0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0;
             0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0;
             0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0;
             0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1];
 
    
 otherwise
  error('[%s.m] unknown tree number', mfilename);
end

nvars = size(t, 1);        % total number of variables
nin = floor((nvars-1)/2);  % max number of inner nodes in 't'
if 2*nin + 1 > nvars
  error('[%s.m] tree error', mfilename);
end

if ~exist('p', 'var')

  % for each inner node there is a distribution to generate the LHS and the
  % RHS
  p = cell(nvars, 1);
  % the distribution for the roots
  p0 = cell(1, length(t0));
  for i = 1:length(t0)
    p0{i} = normalizzze(rand(nsyms(t0(i)), 1));
  end
  for i = 1:nvars
    % for each symbol at inner node 'i' we generate a distribution according
    % to the number of possible symbols for the leaf
    nkids = size(t{i}, 2);
    if nkids > 0
      p{i} = cell(1, nkids);
      if nkids == 1
        error('[%s.m] no implementation for single kid', mfilename);
      elseif nkids == 2
        % create block structure with nsyms(i) many blocks
        for j = 1:nkids
          kid_nsyms = nsyms(t{i}(j));
          init = rand(kid_nsyms, 1);
          dist = zeros(kid_nsyms, nsyms(i));
          offset = ceil(kid_nsyms/nsyms(i));
          for jj=1:nsyms(i)-1
            sel = (1:offset) + (jj-1)*offset;
            dist(sel, jj) = init(sel);
          end
          sel = (nsyms(i)-1)*offset+1;
          dist(sel:end, end) = init(sel:end);
          p{i}{j} = normalizzze(dist);
        end      
      else
        warning('[%s.m] implementation for more than two kids not very good', mfilename);
        for j = 1:nkids
          dist = rand(nsyms(t{i}(j)), nsyms(i));
          dist(1, find(sum(dist, 1) == 0.0)) = 1.0;
          p{i}{j} = normalizzze(dist);
        end
      end
    end
  end
end
atree.t0 = t0;
atree.p0 = p0;
atree.t = t;
atree.p = p;
atree.nsyms = nsyms;
atree.nobs = nobs;
atree.df = degreesoffreedom(atree);
check_forrest(atree);
