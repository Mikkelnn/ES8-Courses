function result = rtcrdivide(a,b)
% RTCRDIVIDE   Division of a curve or curve set with a scalar.
%    RTCRDIVIDE(X,D) if X is a curve or curve set and D is a scalar, the 
%    division of X with D is computed. I.e. the curve or curve set X is 
%    scaled on the y-axis with 1/D.
%
%    Examples of usage:
%      >> g = rtcrdivide(f, 3)
%
%    See also RTCTIMES.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'double'))
    result = a.clone();
    result.scaleY(1/b);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'double'))
    s = size(a);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = a(i,j).clone();
            result(i,j).scaleY(1/b);
        end
    end
else
	error('RTCRDIVIDE - rdivide is not defined for the passed argument list.')
end