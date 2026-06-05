function result = rtcmin(a,b)
% RTCMIN   Minimum of two curves, curve sets or curves and numbers.
%    RTCMIN(X,Y) if X and Y are curves or curve sets, the minimum of X 
%    and Y is computed.
%
%    RTCMIN(X,D) if X is a curve set or curve and D is a scalar, the 
%    minimum  of X and the constant D is computed.
% 
%    RTCMIN(D,Y) if Y is a curve or curve set and D is a scalar, the 
%    minimum of Y and the constant D is computed.
%
%    See also RTCMAX.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

sa = size(a);
sb = size(b);

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    result = CurveMath.min(a,b);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'double'))
    result = CurveMath.min(a,Curve([[0 b 0]]));
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double'))
    result = CurveMath.min(Curve([[0 a 0]]),b);      
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    if (sa(1,1) == sb(1,1) && sa(1,2) == sb(1,2)) 
        for i = (1:sa(1,1))
            for j = (1:sa(1,2))
                result(i,j) = CurveMath.min(a(i,j),b(i,j));
            end
        end
    else
        error('RTCMIN - the matrix dimensions of the input arguments must be equal.')
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'double'))
    for i = (1:sa(1,1))
        for j = (1:sa(1,2))
            result(i,j) = CurveMath.min(a(i,j),Curve([[0 b 0]]));
        end
    end    
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double'))
    for i = (1:sb(1,1))
        for j = (1:sb(1,2))
            result(i,j) = CurveMath.min(Curve([[0 a 0]]),b(i,j));
        end
    end 
else
    error('RTCMIN - min is not defined for the passed argument list.')
end