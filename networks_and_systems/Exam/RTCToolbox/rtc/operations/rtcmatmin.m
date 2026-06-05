function [r rc] = rtcmatmin(a, b, ac, bc)
% RTCMATMIN   Performs the minimum of two matrices
%    RTCMATMIN(A, B) = R results in a matrix R of curves, 
%       A and B are matrices of curves. The resulting
%       matrix of curves is the elementwise minimum.
%       The curves A and B are supposed to be
%       0 at 0.
%
%    RTCMATMIN(A, B, AC, BC) results in [R, RC] where A, B and R are 
%       matrices of curves and AC, BC and RC are matrices of doubles.
%       The resulting matrices of curves represent the elementwise minimum.
%       The implementation of the min-convolution in the toolbox
%       assumes that the curves are 0 at 0. In order to allow for general
%       curves, we use constants AC and BC where A and B are 0 at 0, i.e.
%
%       R(t) + RC = min(A(t) + AC), (B(t) + BC))   
%
%    See also RTCMIN, RTCMATMINCONV.
%
%    Author(s): L. THIELE
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if ~(nargin == 2 || nargin == 3 ||nargin == 4)
    error('RTCMATMIN - exepts 2, 3 or 4 arguments.')
end

if ~(strcmp(class(a(1,1)),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b(1,1)),'ch.ethz.rtc.kernel.Curve'))
    error('RTCMATMIN - not defined for the class of inputs provided.')
end

sa = [size(a,1) size(a(1),1)];
sb = [size(b,1) size(b(1),1)];

if (nargin == 2)
    ac = zeros(sa);
    bc = zeros(sb);
elseif (nargin == 3)
    bc = zeros(sb);
end

if (~all(sa == sb) || ~all(sa == size(ac)) || ~all(sb == size(bc))) 
    error('RTCMATMIN - sizes of provided input matrices do not match.')
end

r(sa(1), sa(2)) = Curve([[0 0 0]]); % initialize Java array
rc(sa(1), sa(2)) = Inf; % initialize array
for i = (1:sa(1))
	for j = (1:sa(2))
            if (ac(i,j) > bc(i,j))
                r(i,j) = rtcmin(rtcscale(a(i,j), 1, ac(i,j) - bc(i,j)), b(i,j));
                rc(i,j) = bc(i,j);
            else
                if isinf(bc(i,j))
                    r(i,j) = Curve([[0 Inf 0]]);
                else
                    r(i,j) = rtcmin(a(i,j) , rtcscale(b(i,j), 1, bc(i,j) - ac(i,j)));
                end
                rc(i,j) = ac(i,j);
            end
    end
end

   