function result = rtch(a,b)
% RTCH   Compute the max horizontal distance betw two curves or curve sets
%    RTCH(X,Y) computes the maximum horizontal distance between the
%    curves or curve sets X and Y.
%
%    The result equals the maximum distance when starting at curve
%    X(upper), and going horizontally to the right until hitting curve 
%    Y(lower). If the long-term slope of X (upper) is bigger than the 
%    long-term slope of Y (lower), then the result is infinity.
%
%    The parameters must fulfill the following condition:
%      - X and Y must be wide-sense increasing 
%
%    See also RTCPLOTH, RTCV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve')) && (strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    try
        result = CurveMath.maxHDist(a,b);
    catch
        error('H(X,Y) - X and Y must be wide-sense increasing curves.')
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]')) && (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    try
        result = CurveMath.maxHDist(a(1),b(2));
    catch
        error('H(X,Y) - X and Y must be wide-sense increasing curves.')
    end
else
    error('RTCH - h not defined for the passed argument list.')
end