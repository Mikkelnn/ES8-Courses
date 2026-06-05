function result = rtcceil(c)
% RTCCEIL   Ceil of a curve or curve set.
%    RTCCEIL(X) computes the ceil of a curve
%    or a curve set X.
%
%    See also RTCFLOOR.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]')
    s = size(c);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = CurveMath.ceil(c(i,j));
        end
    end
elseif strcmp(class(c),'ch.ethz.rtc.kernel.Curve')
    result = CurveMath.ceil(c);
else
    error('RTCCEIL - ceil not defined for the passed argument list.')
end