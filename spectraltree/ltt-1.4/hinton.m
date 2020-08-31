function h = hinton(w, mw);
%HINTON	Plot Hinton diagram for a weight matrix.
%
%	Description
%
%	HINTON(W) takes a matrix W and plots the Hinton diagram.
%
%	H = HINTON(NET) also returns the figure handle H which can be used,
%	for instance, to delete the  figure when it is no longer needed.
%
%	To print the figure correctly in black and white, you should call
%	SET(H, 'INVERTHARDCOPY', 'OFF') before printing.
%
%	See also
%	DEMHINT, HINTMAT, MLPHINT
%

%	Copyright (c) Ian T Nabney (1996-2001)

%%     Copyright (c) 1996-2001, Ian T. Nabney
%%     All rights reserved.
%%
%%     Redistribution and use in source and binary
%%     forms, with or without modification, are
%%     permitted provided that the following
%%     conditions are met:
%%
%%	  * Redistributions of source code must
%%	    retain the above copyright notice, this
%%	    list of conditions and the following
%%	    disclaimer.
%%	  * Redistributions in binary form must
%%	    reproduce the above copyright notice,
%%	    this list of conditions and the
%%	    following disclaimer in the
%%	    documentation and/or other materials
%%	    provided with the distribution.
%%	  * Neither the name of the Aston University, Birmingham, U.K.
%%	    nor the names of its contributors may be
%%	    used to endorse or promote products
%%	    derived from this software without
%%	    specific prior written permission.
%%
%%     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT
%%     HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
%%     EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
%%     NOT LIMITED TO, THE IMPLIED WARRANTIES OF
%%     MERCHANTABILITY AND FITNESS FOR A PARTICULAR
%%     PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
%%     REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
%%     DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
%%     EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%%     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
%%     OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
%%     DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
%%     HOWEVER CAUSED AND ON ANY THEORY OF
%%     LIABILITY, WHETHER IN CONTRACT, STRICT
%%     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
%%     OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%%     OF THIS SOFTWARE, EVEN IF ADVISED OF THE
%%     POSSIBILITY OF SUCH DAMAGE.
      
      
% Set scale to be up to 0.9 of maximum absolute weight value, where scale
% defined so that area of box proportional to weight value.

if ~exist('mv', 'var'), mv = []; end

% Use no more than 640x480 pixels
xmax = 640; ymax = 480;

% Offset bottom left hand corner
x01 = 40; y01 = 40;
x02 = 80; y02 = 80;

% Need to allow 5 pixels border for window frame: but 30 at top
border = 5;
top_border = 30;

ymax = ymax - top_border;
xmax = xmax - border;

% First layer

[xvals, yvals, color] = hintmat(w, mw);
% Try to preserve aspect ratio approximately
if (8*size(w, 1) < 6*size(w, 2))
  delx = xmax; dely = xmax*size(w, 1)/(size(w, 2));
else
  delx = ymax*size(w, 2)/size(w, 1); dely = ymax;
end


h = figure('Color', [0.5 0.5 0.5], ...
           'Name', 'Hinton diagram', ...
           'NumberTitle', 'off', ...
           'Units', 'pixels', ...
           'Position', [x01 y01 delx dely]);
set(gca, 'Visible', 'off', 'Position', [0 0 1 1]);
patch(xvals', yvals', 0.3*color', 'Edgecolor', 'none');
axis equal;
