function result = rtcscale(c, a, b)
% RTCSCALE   An affine value transformation of a curve or curve set.
%    RTCSCALE(C,A,B) if C is a curve or curve set, B is a scalar, and A
%      is a scalar, then A*C(x)+B is computed.
%      Note that the curve may get negative, dependent on the values of 
%      A and B.
%
%    Examples of usage:
%      >> g = rtcscale(f, 2, -3)
%
%    See also RTCAFFINE, RTCTIMES.

%    Author(s): L. Thiele
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double') && strcmp(class(b),'double'))
    s = size(c);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = c(i,j).clone();
            result(i,j).scaleY(a);
            result(i,j).move(0,b);
        end
    end     
elseif (strcmp(class(c),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double') && strcmp(class(b),'double'))
    result = c.clone();
    result.scaleY(a);
    result.move(0,b);
else
    error('RTCSCALE - is not defined for the passed argument list.')
end


