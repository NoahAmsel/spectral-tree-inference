% ltt_demo1.m
% demonstrates ltt on dataset 1 "binary forest"
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).
close all
clear
opt.verbose = 1;  % for demonstrations
dataset = 1;
msg = sprintf('%d "BINARY-FOREST", which is generated by the following tree\n', dataset);
msg = sprintf('%sstructure:\n', msg);
msg = sprintf('%s\n', msg');
msg = sprintf('%s%s\n', msg, '            7          8  ');
msg = sprintf('%s%s\n', msg, '           / \        / \ ');
msg = sprintf('%s%s\n', msg, '          1   6      4   5');
msg = sprintf('%s%s\n', msg, '             / \          ');
msg = sprintf('%s%s\n', msg, '            2   3         ');
msg = sprintf('%s\n', msg);
msg = sprintf('%sThe leaf variables x2 and x3 have eight states.  The leaves x1, x4, x5\n', msg);
msg = sprintf('%sand inner node x6 have four states, and the roots x7 and x8 have two\n', msg);
msg = sprintf('%sstates.  The root states are chosen by fair coin flips. For\n', msg);
msg = sprintf('%seach state that a parent node can take on, there are two possible\n', msg);
msg = sprintf('%sstates for a child, selected with a fair coin flip.\n', msg);
msg = sprintf('%sFor more details on the CPTs please take a look at "create_tree.m".\n', msg);

msg_final = sprintf('Note that the correct solution has been found.\n');

% (0) introduction
clc
fprintf('[%s.m] starting demonstration\n', mfilename);
ltt_demo_helper
