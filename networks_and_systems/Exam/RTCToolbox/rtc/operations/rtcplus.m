function result = rtcplus(a,b)
% RTCPLUS   Addition of two curves or curve sets.
%    RTCPLUS(X,Y) if X and Y are curves or curve sets, the addition of 
%    X and Y is computed.
%
%    RTCPLUS(X,D) if X is a curve or curve set and D is a scalar, the 
%    addition of X and the constant D is computed.
% 
%    RTCPLUS(D,Y) if Y is a curve or curve set and D is a scalar, the 
%    addition of Y and the constant D is computed.
%
%    Examples of usage:
%      >> g = rtcplus(f, g)
%      >> g = rtcplus(f, 4)
%      >> g = rtcplus(3, g)
%
%    See also RTCMINUS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

sa = size(a);
sb = size(b);

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    result = CurveMath.add(a,b);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'double'))
	result = a.clone();
	result.move(0,b);
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double'))
   	result = b.clone();
	result.move(0,a);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    if (sa(1,1) == sb(1,1) && sa(1,2) == sb(1,2))
        for i = (1:sa(1,1))
            for j = (1:sa(1,2))
                result(i,j) = CurveMath.add(a(i,j),b(i,j));
            end
        end
    else
        error('RTCADD - the matrix dimensions of the input arguments must be equal.')
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'double'))
    for i = (1:sa(1,1))
        for j = (1:sa(1,2))
            result(i,j) = a(i,j).clone();
            result(i,j).move(0,b);
        end
    end
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double'))
    for i = (1:sb(1,1))
        for j = (1:sb(1,2))
            result(i,j) = b(i,j).clone();
            result(i,j).move(0,a);
        end
    end
else
    error('RTCPLUS - minus is not defined for the passed argument list.')
end