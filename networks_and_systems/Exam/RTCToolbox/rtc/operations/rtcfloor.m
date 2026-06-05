function result = rtcfloor(c)
% RTCFLOOR   Floor of a curve or curve set.
%    RTCFLOOR(X) computes the floor of curve 
%    or curve set X.
%
%    See also RTCCEIL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]')
    s = size(c);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = CurveMath.floor(c(i,j));
        end
    end
elseif strcmp(class(c),'ch.ethz.rtc.kernel.Curve')
    result = CurveMath.floor(c);
else
    error('RTCFLOOR - floor not defined for the passed argument list.')
end