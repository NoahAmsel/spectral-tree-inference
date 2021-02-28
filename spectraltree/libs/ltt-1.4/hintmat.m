function [xvals, yvals, color] = hintmat(w, mw);
%HINTMAT Evaluates the coordinates of the patches for a Hinton diagram.
%
%	Description
%	[xvals, yvals, color] = hintmat(w)
%	  takes a matrix W and returns coordinates XVALS, YVALS for the
%	patches comrising the Hinton diagram, together with a vector COLOR
%	labelling the color (black or white) of the corresponding elements
%	according to their sign.
%
%	See also
%	HINTON
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

w = flipud(w);
[nrows, ncols] = size(w);

if ~exist('mw','var') || isempty(mw)
  mw = max(max(abs(w)));            % ADDITION BY S.H.
end
scale = 0.45*sqrt(abs(w)/mw);
scale = scale(:);
color = 0.5*(sign(w(:)) + 3);

delx = 1;
dely = 1;
[X, Y] = meshgrid(0.5*delx:delx:(ncols-0.5*delx), 0.5*dely:dely:(nrows-0.5*dely));

% Now convert from matrix format to column vector format, and then duplicate
% columns with appropriate offsets determined by normalized weight magnitudes. 

xtemp = X(:);
ytemp = Y(:);

xvals = [xtemp-delx*scale, xtemp+delx*scale, ...
         xtemp+delx*scale, xtemp-delx*scale];
yvals = [ytemp-dely*scale, ytemp-dely*scale, ...
         ytemp+dely*scale, ytemp+dely*scale];

