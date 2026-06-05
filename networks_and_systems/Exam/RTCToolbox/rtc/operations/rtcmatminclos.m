function [r rc] = rtcmatminclos(a, ac, n)
% RTCMATMINCLOS   Performs the min-closure of the input matrix
%    RTCMATMINCLOS(A, AC, n) = [R RC] where A and R are 
%       matrices of curves and AC, RC are matrices of doubles.
%       The result of RTCMATMINCLOS is
%       (R + RC) = E ^ (A + AC) ^ (A + AC)^2 ^ ... ^ (A + AC)^m
%       where A and R are curves which are 0 at 0, ^ is the minimum
%       operator, multiplication corresponds to min-convolution,
%       and m = 2^n . E is a matrix with infinite elements, only the
%       diagonals are 0 at 0 and infinite otherwise.
%
%    See also RTCMIN, RTCMATMINCONV.
%
%    Author(s): L. THIELE
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if ~(nargin == 3)
    error('RTCMATMINCLOS - exepts 2, 3 or 4 arguments.')
end

if ~(strcmp(class(a(1,1)),'ch.ethz.rtc.kernel.Curve'))
    error('RTCMATMINCLOS - not defined for the class of inputs provided.')
end

sa = [size(a,1) size(a(1),1)];

if (~all(sa == size(ac))) 
    error('RTCMATMINCLOS - sizes of provided input matrices do not match.')
end

% determine initial matrix (R + RC) = E ^ (A + AC)
r(sa(1), sa(2)) = Curve([[0 0 0]]); % initialize Java array
rc(sa(1), sa(2)) = 0; % initialize array
for i = (1:sa(1))
	for j = (1:sa(2))
        r(i,j) = a(i,j);
        rc(i,j) = ac(i,j);
    end
    r(i,i) = rtcscale(a(i,i), 1, ac(i,i));
    rc(i,i) = 0;
end

% iterate n times
for i = (1:n)
    [r rc] = rtcmatminconv(r, r, rc, rc);
end

   