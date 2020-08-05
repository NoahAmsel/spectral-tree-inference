% ltt_demo4.m
% demonstrates ltt on dataset 4 "six-coins"
% Copyright (C) 2006 - 2009 by Stefan Harmeling (2009-06-26).
close all
clear
opt.verbose = 1;  % for demonstrations
dataset = 4;
msg = sprintf('%d "SIX-COINS", which is generated by the\n', dataset);
msg = sprintf('%sfollowing tree structure:\n', msg);
msg = sprintf('%s\n', msg');
msg = sprintf('%s%s\n', msg, '          6        ');
msg = sprintf('%s%s\n', msg, '        / | \      ');
msg = sprintf('%s%s\n', msg, '       1  2  5     ');
msg = sprintf('%s%s\n', msg, '            / \    ');
msg = sprintf('%s%s\n', msg, '           3   4   ');
msg = sprintf('%s\n', msg);
msg = sprintf('%sThe leaf variables x1, x2, x3, x4 have each eight states.  The inner nodes x5\n', msg);
msg = sprintf('%sand x6 each have sixteen states.  The distributions can be understood\n', msg);
msg = sprintf('%sas follows:  suppose we flip six coins a1, a2, a3, a4, a5, a6 (with\n', msg);
msg = sprintf('%svalues zero and one).  Then let\n', msg);
msg = sprintf('%s\n', msg);
msg = sprintf('%s   x6 = a1 + 2*a2 + 4*a3 + 8*a4\n', msg);
msg = sprintf('%s   x2 = a1 + 2*a2 + 4*a4\n', msg);
msg = sprintf('%s   x1 = a1 + 2*a2 + 4*a3\n', msg);
msg = sprintf('%s   x5 = a3 + 2*a4 + 4*a5 + 8*a6\n', msg);
msg = sprintf('%s   x4 = a3 + 2*a4 + 4*a6\n', msg);
msg = sprintf('%s   x3 = a3 + 2*a4 + 4*a5\n', msg);
msg = sprintf('%s\n', msg);
msg = sprintf('%sIn this situation x1 and x2 share two bits, x3 and x4 share two bits\n', msg);
msg = sprintf('%sas well, while all other pairs of leaves share a single bit.\n', msg);

msg_final = sprintf('The SIX-COINS data has been designed to make our algorithm fail.\n');
msg_final = sprintf('%sAs expected it first merged x1 and x2 and also x3 and x4,\n', msg_final);
msg_final = sprintf('%ssince both pair share two bits.  However, the inferred latent\n', msg_final);
msg_final = sprintf('%snodes do not encode the other bits that are shared across the\n', msg_final);
msg_final = sprintf('%stwo groups, thus the latent nodes have nothing in common and\n', msg_final);
msg_final = sprintf('%sare not merged which is exactly what the algorithm is supposed\n', msg_final);
msg_final = sprintf('%sto do.\n', msg_final);

% (0) introduction
clc
fprintf('[%s.m] starting demonstration\n', mfilename);
ltt_demo_helper
