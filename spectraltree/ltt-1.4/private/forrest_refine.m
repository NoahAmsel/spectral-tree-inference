function t = forrest_refine(data, t);
% refine a given tree, i.e. improve its cpts
%
% more precisely, we do:
%
%   1. bottom-up propagation to generate all beta
%   2. top-down propagation to generate all pi
%   3. update the CPTS
%
% Stefan Harmeling, June 2008

data.n = size(data.x, 2);

t.beta = cell(7,1);
for j=1:length(t.t0)  % loop over all roots
  t = gen_beta(t.t0(j), data, t);
end
t.pie = cell(7,1);
for j=1:length(t.t0)  % loop over all roots
  t.pie{t.t0(j)} = t.p0{j} * ones(1, data.n);
  t = gen_pie(t.t0(j), data, t);
end
for j=1:length(t.t0)  % loop over all roots
  t.p0{j} = normalizzze(sum(normalizzze(t.beta{t.t0(j)} .* t.pie{t.t0(j)}), 2));
  t = upd_cpts(t.t0(j), data, t);
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function atree = gen_beta(subtree, data, atree);
% do we have data at the current node?
if subtree > atree.nobs
  % NOT OBSERVED
  % create a vector with ones that allows all values
  msg = ones(atree.nsyms(subtree), data.n);
else
  % OBSERVED
  % create a vector with zeros and a single one, which will pick out the
  % correct row from the CPT of the parent
  msg = zeros(atree.nsyms(subtree), data.n);
  for i=1:data.n
    msg(data.x(subtree,i), i) = 1;
  end
end

% does subtree has kids?
nkids = size(atree.t{subtree}, 2);
if nkids > 0
  % ask for the messages of the kids
  for j = 1:nkids
    kid = atree.t{subtree}(j);
    atree = gen_beta(kid, data, atree);
    cpt = atree.p{subtree}{j};
    % the matrix product does sum out the kid
    kid_msg = atree.beta{kid};
    msg = msg .* (cpt' * kid_msg);
  end
end
atree.beta{subtree} = msg;
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function atree = gen_pie(subtree, data, atree);
% when this function is called 'subtree' already has its pie!

% does subtree has kids?
nkids = size(atree.t{subtree}, 2);
if nkids > 0
  % generate pie for the kids
  for j = 1:nkids
    % multiply all beta messages of the other kids:
    msg = ones(atree.nsyms(subtree), data.n);
    for jj = 1:nkids
      if j ~= jj
        kid = atree.t{subtree}(jj);
        kid_msg = atree.beta{kid};
        cpt = atree.p{subtree}{jj};
        msg = msg .* (cpt' * kid_msg);
      end
    end
    kid = atree.t{subtree}(j);
    msg = atree.pie{subtree} .* msg;
    atree.pie_msg{kid} = msg;
    cpt = atree.p{subtree}{j};
    atree.pie{kid} = normalizzze(cpt * msg);
    % go down the tree
    atree = gen_pie(kid, data, atree);
  end
end
return


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function atree = upd_cpts(subtree, data, atree);
% when this function is called there should be 'beta' and 'pie'

% does subtree has kids?
nkids = size(atree.t{subtree}, 2);
if nkids > 0
  % update the CPTs of the kids
  for j = 1:nkids
    kid = atree.t{subtree}(j);
    pie_msg = atree.pie_msg{kid};  % the lambda msg received from subtree
    beta_msg = atree.beta{kid};  % its beta msg
    cpt = atree.p{subtree}{j};
    msgs = zeros(size(cpt));
    for i = 1:data.n
      msg = cpt .* (beta_msg(:,i) * pie_msg(:,i)');
      msg = msg/sum(msg(:));    % normalize joint distribution
      % now msg == q(i,k) see Eq. (40) in note.pdf
      msgs = msgs + msg;
    end
    msgs = normalizzze(msgs);   % normalize wrt to first variable
    atree.p{subtree}{j} = msgs;
    
    % go down the tree
    atree = upd_cpts(kid, data, atree);
  end
end
return
