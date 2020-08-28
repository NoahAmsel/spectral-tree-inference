% ltt_demo_helper.m
% helper script for "ltt_demo?.m"
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).

% (0) introduction
disp(' ');
disp('This program demonstrates the algorithm "BIN" for greedy learning of binary')
fprintf('trees on dataset %s', msg);
disp(' ')
disp('Data is generated from the whole tree, but the algorithm will only see the')
disp('data at the leaves.  From that it will try to reconstruct the tree structure.')
disp(' ')
disp('Press any key to generate the data.')
pause
clc

% (1) generate data
% 1  binary-forest
[dtr, dte, t] = create_data(dataset, opt.verbose);
figure(1)
clf
subplot(121)
tree2fig(t)       % pure matlab
%%% tree2dot(t)   % might work but requires GraphViz
title('true structure')
disp(' ')
disp('The left panel of Figure 1 shows the true tree structure.  Each node is')
disp('labelled with an unique integer and in brackets with the number of states.')
disp(' ')
disp('Press any key to run algorithm BIN on the data of the leaves.')
pause
clc

% (2) call method
% 'bin'    binary trees (the proposed method in our paper)
method = 'bin';
t_hat = ltt(method, dtr, opt);
  
% (3) finish
if exist('msg_final', 'var')
  fprintf('\n%s', msg_final);
end
disp(' ')
disp('The demonstration is now complete.  Please press any key to exit.');
pause
