function result = rtctimes(a,b)
% RTCTIMES   Multiplication of a curve or curve set with a scalar.
%    RTCTIMES(X,D) if X is a curve or curve set and D is a scalar, the 
%    multiplication of X with D is computed. I.e. curve or curve set X is 
%    scaled on the y-axis with D.
% 
%    RTCTIMES(D,Y) if Y is a curve or curve set and D is a scalar, the 
%    multiplication of Y with D is computed. I.e. curve or curve set Y is 
%    scaled on the y-axis with D.
%
%    Examples of usage:
%      >> g = rtctimes(f, 3)
%
%    See also RTCRDIVIDE.
%
%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'double'))
    s = size(a);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = a(i,j).clone();
            result(i,j).scaleY(b);
        end
    end    
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double'))
    s = size(b);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = b(i,j).clone();
            result(i,j).scaleY(a);
        end
    end    
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'double'))
    result = a.clone();
    result.scaleY(b);
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double'))
    result = b.clone();
    result.scaleY(a);
else
    error('RTCTIMES - times is not defined for the passed argument list.')
end