function result = rtcuminus(c)
% RTCUMINUS   Unary minus of a curve or curve set.
%    RTCUMINUS(X) computes the negation of curve or curve set X.
%
%    Examples of usage:
%      >> g = rtcuminus(f)
%
%    See also RTCUPLUS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]'))
    s = size(c);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = c(i,j).clone();
            result(i,j).scaleY(-1);
        end
    end    
elseif (strcmp(class(c),'ch.ethz.rtc.kernel.Curve'))
	result = c.clone();
    result.scaleY(-1);
else
    error('RTCUMINUS - uminus is not defined for the passed argument list.')
end