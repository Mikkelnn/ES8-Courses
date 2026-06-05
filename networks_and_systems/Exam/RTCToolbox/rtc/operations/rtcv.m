function result = rtcv(a,b)
% RTCV   Compute the max vertical distance bet. two curves or curve sets.
%    RTCV(X,Y) computes the maximum vertical distance between the
%    curves or curve sets X and Y.
%
%    The result equals the maximum y-value of the curve X(upper)-Y(lower). 
%    If the long-term slope of X(upper) is bigger than the one of Y(lower), 
%    then the result is infinity.
%
%    See also RTCPLOTV, RTCH.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    result = CurveMath.maxVDist(a(1),b(2));
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    result = CurveMath.maxVDist(a,b);
else
    error('RTCV - v not defined for the passed argument list.')
end

