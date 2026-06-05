function [r rc] = rtcmatminconv(a, b, ac, bc)
% RTCMATMINCONV   Performs the min-convolution of two matrices
%    RTCMATMINCONV(A, B) = R results in a matrix of curves R, 
%       A and B are matrices of curves.
%       The min-convolution of the two matrices A and B is
%       defined like the multiplication of two matrices, but 
%       multiplication is min-convolution and addition is infimum:
%       Result = A \otimes B. The curves A and B are supposed to be
%       0 at 0.
%
%    RTCMATMINCONV(A, B, AC, BC) results in (R, RC) where A, B and R are 
%       matrices of curves and AC, BC and RC are matrices of doubles.
%       The min-convolution of the two matrices A and B is
%       defined like the multiplication of two matrices, but 
%       multiplication is min-convolution and addition is infimum.
%       The implementation of the min-convolution in the toolbox
%       assumes that the curves are 0 at 0. In order to allow for general
%       curves, we use constants AC and BC where A and B are 0 at 0. 
%       The result of RTCMATMINCONV is (R, RC) where R(0) = 0 and 
%
%       R(t) + RC = (A(t) + AC) \otimes (B(t) + BC)   
%
%    See also RTCMINCONV.
%
%    Author(s): L. THIELE
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if ~(nargin == 2 || nargin == 3 ||nargin == 4)
    error('RTCMATMINCONV - exepts 2, 3 or 4 arguments.')
end

if ~(strcmp(class(a(1,1)),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b(1,1)),'ch.ethz.rtc.kernel.Curve'))
    error('RTCMATMINCONV - not defined for the class of inputs provided.')
end

sa = [size(a,1) size(a(1),1)];
sb = [size(b,1) size(b(1),1)];

if (nargin == 2)
    ac = zeros(sa);
    bc = zeros(sb);
elseif (nargin == 3)
    bc = zeros(sb);
end

if ((sa(2) ~= sb(1)) || ~all(sa == size(ac)) || ~all(sb == size(bc))) 
    error('RTCMATMINCONV - sizes of provided input matrices do not match.')
end

r(sa(1), sb(2)) = Curve([[0 0 0]]); % initialize Java array
rc(sa(1), sb(2)) = Inf; % initialize array

for i = (1:sa(1))
	for j = (1:sb(2))
        tmpA = Curve([[0 Inf 0]]);
        tmpB = Inf;
        for k = (1:sa(2))
            tmp1 = rtcminconv(a(i,k), b(k,j)); %the conv of the two elements
            tmp2 = ac(i,k) + bc(k,j); %the combined constant of the two elements
            if (tmpB > tmp2) %the old initial value is larger than the new
                tmpA = rtcmin(rtcscale(tmpA, 1, tmpB - tmp2), tmp1);
                tmpB = tmp2; %the new inital value is the smaller one
            else
                if ~isinf(tmp2)
                    tmpA = rtcmin(tmpA , rtcscale(tmp1, 1, tmp2 - tmpB));
                end
            end
        end
        r(i,j) = tmpA; rc(i,j) = tmpB;
    end
end

   