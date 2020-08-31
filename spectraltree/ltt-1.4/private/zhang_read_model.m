function atree = zhang_read_model(fname);
% utility function for "zhang_forrest.m".
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

t0 = [];   % the roots
p0 = {};
t = {};
p = [];

% (1) extract all roots, all latent and all observed variables
all = {};
latent = {};
children = {};
fid = fopen(fname);
while 1
  tline = fgetl(fid);
  if ~ischar(tline), break, end
  if length(tline) > 7 && strcmp('variable', tline(1:8))
    re = regexp(tline, '"(\w+)"', 'tokens');
    all{end+1} = re{1}{1};
  elseif length(tline) > 10 && strcmp('probability', tline(1:11))
    re = regexp(tline, '"(\w+)"', 'tokens');
    children{end+1} = re{1}{1};
    if length(re) > 1
      latent{end+1} = re{2}{1};      
    end
  elseif length(tline) > 31 && ...
        strcmp('//LoglikelihoodOfPreviousModel:', tline(1:31))
    ll = sscanf(tline(32:end), '%f');
  end
end
fclose(fid);
all = unique(all);
children = unique(children);
latent = unique(latent);
roots = setdiff(all, children);
observed = setdiff(all, latent);
all = [sort(observed), sort(latent)];  % fix this ordering

lroots = length(roots);
lall = length(all);
t0 = [];
p0 = {};
t = cell(lall, 1);
p = cell(lall, 1);

% (2) extract all nsyms
nsyms = zeros(1, lall);
fid = fopen(fname);
while 1
  tline = fgetl(fid);
  if ~ischar(tline), break, end
  if length(tline) > 7 && strcmp('variable', tline(1:8))
    re = regexp(tline, '"(\w+)"', 'tokens');
    [tf, left] = ismember(re{1}{1}, all);  % get index of the variable
    tline = fgetl(fid);
    re = regexp(tline, '"(\w+)"', 'tokens');
    nsyms(left) = length(re);
  end
end
fclose(fid);


% (3) extract all CPTs
fid = fopen(fname);
while 1
  tline = fgetl(fid);
  if ~ischar(tline), break, end
  if length(tline) > 10 && strcmp('probability', tline(1:11))
    re = regexp(tline, '"(\w+)"', 'tokens');
    left = re{1}{1};
    [tf, left] = ismember(left, all);  % get index of left variable
    if length(re) > 1
      right = re{2}{1};
      [tf, right] = ismember(right, all);  % get index of right variable
      t{right}(end+1) = left;
      p{right}{end+1} = zeros(nsyms(left), nsyms(right));
      for i = 1:nsyms(right)
        tline = fgetl(fid);
        while ismember(tline(15), [ ')', '"' ])
          tline = tline(2:end);
        end
        p{right}{end}(:, i) = sscanf(tline(15:end-1), '%f');
      end
    else
      % left must be root
      t0 = [t0 left];
      tline = fgetl(fid);
      row = sscanf(tline(12:end-1), '%f');
      p0{end+1} = row;
    end
  end
end
fclose(fid);

% (4) collect all information
atree.t = t;
atree.p = p;
atree.t0 = t0;
atree.p0 = p0;
atree.nsyms = nsyms;
atree.nobs = length(observed);
atree.ll_zhang = ll;
atree.varnames = all;
atree.df = degreesoffreedom(atree);
check_forrest(atree);