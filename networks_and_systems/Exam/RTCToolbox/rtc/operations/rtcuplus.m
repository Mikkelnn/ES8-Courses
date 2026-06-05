function result = rtcuplus(c)
% RTCUPLUS   Unary plus of a curve or curve set creates a clone.
%    RTCUPLUS(X) does not compute anything, but creates a clone of the
%    curve or curve set X. 
%
%    Examples of usage:
%      >> g = rtcuplus(f)
%
%    See also RTCUMINUS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: uplus.m 329 2006-02-17 09:57:08Z wandeler $

import ch.ethz.rtc.kernel.*;
if (strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]'))
    s = size(c);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = c(i,j).clone();
        end
    end   
elseif (strcmp(class(c),'ch.ethz.rtc.kernel.Curve'))
	result = c.clone();
else
    error('RTCUPLUS - uplus is not defined for the passed argument list.')
end