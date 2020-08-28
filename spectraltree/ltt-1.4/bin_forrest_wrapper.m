## Copyright (C) 2020 yariv
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} bin_forrest_wrapper (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: yariv <yariv@LAPTOP-7VMUDEEC>
## Created: 2020-07-29

function adj_mat = bin_forrest_wrapper (data,m)
% data - num_of_leafs X N matrix
% m - number of letters in the alphabet
if ~exist('m', 'var')
  m=4; 
end
data_struct.x = data;
data_struct.nsyms = m * ones(1,size(data)(1));
data_struct.name = "wrapper";

atree = bin_forrest(data_struct);

adj_mat = zeros(length(atree.t),length(atree.t));
for ind = 1:length(atree.t)
  for int_ind = 1:length(atree.t{ind})
    adj_mat(ind,atree.t{ind}(int_ind)) = 1;
    adj_mat(atree.t{ind}(int_ind), ind) = 1;
  end
end
endfunction
