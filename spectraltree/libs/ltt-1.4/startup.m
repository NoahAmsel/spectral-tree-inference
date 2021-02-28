% add the path to the datasets
addpath('data');

if ~exist('normalizzze')
  disp('compiling normalizzze.c')
  mex normalizzze.c
end
