function atree = zhang_forrest(d);
% Calls Zhang's JAVA code.
%
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

% SET THE FOLLOWING PATHS CORRECTLY:
% path to Zhang's java code
CODEPATH = '/kyb/agbs/harmeling/Projects/pcfg/lib/hlcm-distribute';
% path where intermediate data can be stored
DATAPATH = '/kyb/agbs/harmeling/cluster/pcfg/zhang_data';

% convert the data to Zhang's format
zhang_input= data2zhang(d);

% go to the java code
current = pwd;
cd(CODEPATH);

% generate a directory for the intermediate data
thebase = fullfile(DATAPATH, d.name, '');
if ~exist(thebase, 'dir')
  mkdir(thebase);
end
% generate (more or less unique) dirname
counter = 0;
while 1
  thedir = fullfile(thebase, [d.name sprintf('_%08.0f', counter)]);
  if ~exist(thedir, 'dir')
    mkdir(thedir);
    break
  end
  counter = counter + 1;
  if counter>10000
    error('[%s.m] can not generate unique dirname', mfilename);
  end
end
thefile = fullfile(thedir, 'input.txt');

% transform the data to Zhang's format
fid = fopen(thefile, 'w');
fprintf(fid, '%s', zhang_input);
fclose(fid);

% before running the JAVA code
cmd = ['java -Xmx2000M -cp colt.jar:. LearnHLCM 10 ' thefile ' ' thedir ' BIC'];
fprintf('[%s.m] %s\n', mfilename, cmd);
[status, result] = system(cmd);
cd(current);
if status > 0
  % some error happened
  error('[%s.m] JAVA crashed with the following message:\n %s\n', mfilename, result);
end
atree = zhang_read_model(fullfile(thedir, 'M.BIC.txt'));
atree.name = [d.name '_zhang'];
