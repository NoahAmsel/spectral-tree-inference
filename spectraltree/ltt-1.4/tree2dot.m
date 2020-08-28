function tree2dot(atree, fname, fmt, hintonplots, verbose);
% generates a PDF using the Graphviz software (not included, see
% http://www.graphviz.org/).
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') | isempty(verbose)
  verbose = 0;
end

engine = 'dot';
if ~exist('fname', 'var') | isempty(fname)
  fname = fullfile('results', atree.name);
  if ~exist('results', 'dir')
    mkdir('results');
  end
end
fid = fopen([fname '.dot'], 'w');
gname = atree.name;
gname(gname=='-') = '_';

if ~exist('hintonplots', 'var')
  if any(atree.nsyms ~= 2)
    hintonplots = 0;
  else
    hintonplots = 1;    % with Hinton plots
    hintonplots = 0;
  end
end

hinton_fname = @(fname, i) sprintf('tmp/hinton_%s_%04d.png', fname, i);
if hintonplots == 1
  if ~exist('tmp', 'dir')
    mkdir('tmp');
  end
  switch 2
   case 1  
    % for roots show:
    %   p(x1=0)
    %   p(x1=1)
    t0 = atree.t0;
    p0 = atree.p0;
    for i = 1:length(t0)
      root = t0(i);
      w = p0{i};
      make_hinton_plot(w, hinton_fname(fname, root));
    end
    % and the children show
    % p(x2=0 | x1=0)   p(x2=0 | x1=1)
    % p(x2=1 | x1=0)   p(x2=1 | x1=1)
    t = atree.t;
    p = atree.p;
    for i = 1:size(p, 1)
      % produce the Hinton plots of the kids
      for j = 1:length(t{i})
        kid = t{i}(j);    % index of the child
        w = p{i}{j};
        make_hinton_plot(w, hinton_fname(fname, kid));
      end
    end
    
   case 2
    % for latent variables
    % p(x2=1 | x1=0)   p(x3=1 | x1=0)
    % p(x2=1 | x1=1)   p(x3=1 | x1=1)
    t = atree.t;
    p = atree.p;
    for i = 1:size(p, 1)
      switch length(t{i})
       case 2
        w = [p{i}{1}(2,:)', p{i}{2}(2,:)'];
        make_hinton_plot(w, hinton_fname(fname, i));
       case 0
        % do nothing
       otherwise
        error('[%s.m] can do the Hinton plots only for binary trees', mfilename);
      end
    end
  end
end
switch engine
 case 'dot'
  str = sprintf('digraph %s {\n', gname);
  t = atree.t;
  for i = 1:size(t, 1)
    % set labels for all nodes
    if isfield(atree, 'names') && i <= length(atree.names)
      label = atree.names{i};
    else
      if all(atree.nsyms == 2)
        label = sprintf('%d', i);
      else
        label = sprintf('%d(%d)', i, atree.nsyms(i));
      end
    end
    %%%% IF YOU WANT LOCAL MARGINALS UNCOMMENT THE FOLLOWING THREE LINES
    %if isfield(atree, 'pm')  % show marginals
    %      label = sprintf('%s\\n%f', label, atree.pm{i}(1));
    %    end

    %%%% IF YOU WANT LOCAL BIC DIFFERENCES UNCOMMENT THE FOLLOWING THREE LINES
    %if isfield(atree, 'localbic_diff') && i > atree.nobs
    %  label = sprintf('%s\\n%.2f', label, atree.localbic_diff(i));
    %end
    
    mod = sprintf('label="%s"', label);
    mod = sprintf('%s, shape=box', mod);
    %%%% IF YOU WANT LOCAL MARGINALS UNCOMMENT THE FOLLOWING FOUR LINES
    %if isfield(atree, 'pm')  % show marginals
    %  col = (0.3+0.7*atree.pm{i}(1)) ^2;
    %  mod = sprintf('%s, style=filled, fillcolor=".0 .0 %f"', mod, col);
    %end
    if hintonplots == 1
      hfname = hinton_fname(fname, i);
      if exist(hfname, 'file')
        mod = sprintf('%s, image="%s"', mod, hfname);
      end
    end
    str = sprintf('%s\t%d [%s];\n', str, i, mod);
    for j = 1:size(t{i}, 2)
      % an edge from i to its j-th kid
      str = [str sprintf('\t%d->%d;\n', i, t{i}(j))];
    end
  end
  str = [str '}'];
 case 'neato'
  str = sprintf('digraph %s {\n', gname);
  t = atree.t;
  for i = 1:size(t, 1)
    if isfield(atree, 'names') && i <= length(atree.names)
      str = [str sprintf('\t%d [label="%s"];\n', i, atree.names{i})];
    else
      str = [str sprintf('\t%d [label="%d(%d)"];\n', i, i, atree.nsyms(i))];
    end
    for j = 1:size(t{i}, 2)
      % an edge from i to its j-th kid
      str = [str sprintf('\t%d->%d;\n', i, t{i}(j))];
    end
  end
  str = [str '}'];
 otherwise
  error('[%s.m] unknown engine', mfilename);
end  

fid = fopen([fname '.dot'], 'w');
fprintf(fid, '%s\n', str);
fclose(fid);

%% CALL TO GRAPHVIZ, ALTERNATIVELY CALL BY HAND IN SHELL

if ~exist('fmt', 'var') | isempty(fmt)
  fmt = 'png';   %% one of 'png', 'ps', and on some machines 'pdf'
  % see "man dot" for all available formats on your machine
end
cmd = sprintf('%s -T%s -o %s.%s %s.dot', engine, fmt, fname, fmt, fname);
if verbose > 1
  fprintf('[%s.m] plotting graph to "%s.%s"\n', mfilename, fname, fmt);
  fprintf('[%s.m] running "%s"\n', mfilename, cmd);
end
system(cmd);
switch fmt
 case 'png'
  imshow(imread([fname '.png']));
 case 'pdf'
  system(sprintf('open %s.pdf', fname));
end
return

function str = theshape(str, p)
if p{1}(1,1) > min(p{1}(1,2:end))
  str = sprintf('<< %s', str);
end
if p{2}(1,1) > min(p{2}(1,2:end))
  str = sprintf('%s >>', str);
end
return

function make_hinton_plot(w, hfname);
hinton(w, 1);  % do the Hinton plot
colormap gray
set(gca, 'CLim', [0, 1])
%set(gca, 'CLim', [0, 4]);
set(gcf, 'paperposition', [0,0,0.3,0.3]);
print(gcf, '-dpng', hfname);
close(gcf);
