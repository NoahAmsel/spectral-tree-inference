function [dtr, dte, t] = create_data(dataset, verbose)
% 
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

if ~exist('verbose', 'var') | isempty(verbose)
  verbose = 0;
end

if verbose > 0
  fprintf('[%s.m] generating dataset %d\n', mfilename, dataset);
end

rand('state', 0);
randn('state', 0);

msg = 'run the script "get_other.sh" in "data"';

t = [];
switch dataset
 case {1,2,3,4} % toy data
  NT = 2;    % number of non-terminal symbols
  T  = 4;    % number of terminal symbols
  N = 1000;  % number of data points
  switch dataset
   case 1  % binary-forest
    treenumber = 1;
    name = 'binary-forest';
    
   case 2  % three-level-binary
    treenumber = 2;
    name = 'three-level-binary';
    
   case 3  % three-coins
    treenumber = 3;
    name = 'three-coins';
    
   case 4  % six-coins
    treenumber = 4;
    name = 'six-coins';
  end
  t = create_tree(treenumber);
  t.name = name;
  dtr = sample_from_forrest(N, t);
  dte = sample_from_forrest(N, t);
  nvars = size(dtr.x, 1);    % the total number of variables
  nn = t.nobs;               % the number of leaves
  dtr.x = dtr.x(1:nn, :);    % keep only the leaves
  dtr.nsyms = dtr.nsyms(1:nn);
  dte.x = dte.x(1:nn, :);    % keep only the leaves
  dte.nsyms = dte.nsyms(1:nn);
  
 case {5,6,7}   % coleman / hiv-test / house-building
  ptrue = load('coleman_hiv_house.txt');
  switch dataset
   case 5  % Coleman data
    name = 'coleman';
    ptrue = ptrue(:, [1:4 5]);
   case 6  % hiv-test
    name = 'hiv-test';
    ptrue = ptrue(:, [1:4 6]);
   case 7  % house-building
    name = 'house-building';
    ptrue = ptrue(:, [1:4 7]);
  end
  freq = ptrue(:, 5);
  N = sum(freq);  % total number of observations
  dtr.x = [];
  for i=1:length(freq)
    dtr.x = [dtr.x, repmat(ptrue(i, 1:4)', [1 freq(i)])];
  end
  dtr.x = dtr.x + 1;
  dtr.nsyms = [2 2 2 2];
  dtr.name = name;
    
 case 8  % hannover-5
  name = 'hannover-5';
  ptrue = load('hannover.txt');
  freq = ptrue(:, 6);
  N = sum(freq);  % total number of observations
  dtr.x = [];
  for i=1:length(freq)
    dtr.x = [dtr.x, repmat(ptrue(i, 1:5)', [1 freq(i)])];
  end
  dtr.x = dtr.x + 1;
  dtr.nsyms = [2 2 2 2 2];
  dtr.name = name;
  
 case 9  % hannover-8
  name = 'hannover_all';
  raw_ptrue = load('hannover_all.txt');
  % convert it to 'ptrue'
  raw_ptrue = raw_ptrue';   % transpose
  raw_ptrue = raw_ptrue(:); % vectorize
  ptrue = zeros(256, 9);
  ptrue(:, 9) = raw_ptrue;
  power2 = [1 2 4 8 16 32 64 128];
  for i = 1:256
    a = i-1;
    for j = 8:-1:1
      bit = floor(a/power2(j));
      ptrue(i,j) = bit;
      a = a - bit * power2(j);
    end
  end
  dtr.x = [];
  for i=1:size(ptrue, 1)
    dtr.x = [dtr.x, repmat(ptrue(i, 1:8)', [1 ptrue(i, 9)])];
  end
  dtr.x = dtr.x + 1;
  dtr.nsyms = [2 2 2 2 2 2 2 2];
  dtr.name = name;

 case 10 % UCI car-evaluation
  dtr.name = 'uci_car';
  %%%fname = '/kyb/agbs/harmeling/prj/pcfg/datasets/uci/car/car.data.numeric';
  fname = 'data/car.data.numeric';
  if ~exist(fname, 'file')
    error(msg);
  end
  dtr.x = load(fname)';
  dtr.nsyms = samples2nsyms(dtr.x);
  
 case 11 % vision (from Christoph Lampert)
  name = 'vision';   % vision_location
  xtr = load('vision_voc_object-daten-train.txt');
  xtr = xtr(:, 2:end)';
  xval = load('vision_voc_object-daten-val.txt');
  dtr.x = [xtr, xval(:, 2:end)'];
  xte = load('vision_voc_object-daten-test.txt');
  dte.x = xte(:, 2:end)';
  dtr.nsyms = 10*ones(1, 20);
  dtr.name = 'vision';
  dte.nsyms = 10*ones(1, 20);
  dte.name = 'vision';
  dtr.names = {'aeroplane', 
               'bicycle', 
               'bird', 
               'boat', 
               'bottle', 
               'bus', 
               'car', 
               'cat', 
               'chair', 
               'cow', 
               'diningtable', 
               'dog', 
               'horse', 
               'motorbike', 
               'person', 
               'pottedplant', 
               'sheep', 
               'sofa', 
               'train', 
               'tvmonitor'};
  
 case 12  % vision_01 (from Christoph Lampert)
  name = 'vision_01';   % vision_existence
  xtr = load('vision_voc_object-daten-train.txt');
  xtr = xtr(:, 2:end)';
  xval = load('vision_voc_object-daten-val.txt');
  dtr.x = [xtr, xval(:, 2:end)'];
  xte = load('vision_voc_object-daten-test.txt');
  dte.x = xte(:, 2:end)';
  dtr.x(:) = (1 - (dtr.x(:) == 10)) + 1;
  dte.x(:) = (1 - (dte.x(:) == 10)) + 1;
  dtr.nsyms = 2*ones(1, 20);
  dtr.name = name;
  dte.nsyms = 2*ones(1, 20);
  dte.name = name;
  dtr.names = {'aeroplane', 
               'bicycle', 
               'bird', 
               'boat', 
               'bottle', 
               'bus', 
               'car', 
               'cat', 
               'chair', 
               'cow', 
               'diningtable', 
               'dog', 
               'horse', 
               'motorbike', 
               'person', 
               'pottedplant', 
               'sheep', 
               'sofa', 
               'train', 
               'tvmonitor'};
 
 case 13  % newsgroups from Sam Roweis
  % downloaded from http://www.cs.toronto.edu/~roweis/data/20news_w100.mat
  name = 'news20w100';
  clear dtr dte
  %%%fname = '/kyb/agbs/harmeling/prj/pcfg/datasets/roweis/20news_w100.mat';
  fname = '20news_w100.mat';
  if ~exist(fname, 'file')
    error(msg);
  end
  load(fname);
  dtr.x = documents + 1;
  dtr.nsyms = 2*ones(1, size(documents, 1));
  dtr.name = name;
  dtr.names = wordlist;
  dte.x = dtr.x;
  dte.nsyms = dtr.nsyms;
  dte.name = dtr.name;

 case 14   % COIL-86
  name = 'coil_86';
  if ~exist('ticdata2000.txt', 'file')
    error(msg);
  end
  dtr.x = load('ticdata2000.txt')';
  dte.x = load('ticeval2000.txt')';
  dte.x(end+1,:) = load('tictgts2000.txt')';
  dtr.x([6:86],:) = dtr.x([6:86],:) + 1;
  dte.x([6:86],:) = dte.x([6:86],:) + 1;
  dtr.nsyms = [41, 10, 6, 6, 10, 10, 10, 6, 10, 10, 8, 10, 10, 10, 10, 10, 10, 10, 10, 6, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 4, 7, 5, 10, 8, 8, 10, 6, 8, 7, 7, 10, 7, 4, 8, 9, 4, 7, 2, 7, 6, 3, 6, 2, 13, 6, 9, 5, 4, 7, 7, 4, 9, 2, 2, 3, 8, 2, 3, 5, 3, 3, 2];
  dte.nsyms = dtr.nsyms;
  dtr.name = name;
  dte.name = name;

 case 15   % COIL-42 (from ZHANG)
  name = 'coil_42';
  if ~exist('coilDataTrain_matlab.txt', 'file')
    error(msg);
  end
  raw = load('coilDataTrain_matlab.txt');
  dtr.x = [];
  for i=1:size(raw, 1)
    dtr.x = [dtr.x, repmat(raw(i, 1:42)', [1 raw(i, 43)])];
  end
  dtr.x = dtr.x + 1;
  raw = load('coilDataTest_matlab.txt');
  dte.x = [];
  for i=1:size(raw, 1)
    dte.x = [dte.x, repmat(raw(i, 1:42)', [1 raw(i, 43)])];
  end
  clear raw
  dte.x = dte.x + 1;
  dtr.nsyms = [4, 9, 6, 3, 2, 3, 3, 2, 3, 2, 3, 2, 4, 5, 2, 2, 2, 5, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2];
  dte.nsyms = dtr.nsyms;
  dtr.name = name;
  dte.name = name;
  
 otherwise
  error('[%s.m] unknown dataset number', mfilename);
end

if ~exist('dte', 'var')
  N = size(dtr.x, 2);
  rp = randperm(N);
  Ntr = ceil(N/2);
  Nte = N - Ntr;
  dte.x = dtr.x(:, rp((Ntr+1):end));
  dtr.x = dtr.x(:, rp(1:Ntr));
  dte.nsyms = dtr.nsyms;
  dte.name = dtr.name;
end
