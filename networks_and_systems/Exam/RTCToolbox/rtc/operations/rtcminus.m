function result = rtcminus(a,b)
% RTCMINUS   Substraction of two curves or curve sets.
%    RTCMINUS(X,Y) if X and Y are curves or curve sets, the 
%    substraction of X and Y is computed.
%
%    RTCMINUS(X,D) if X is a curve or curve set and D is a scalar, 
%    the substraction of X and the constant D is computed.
% 
%    RTCMINUS(D,Y) if Y is a curve or curve set and D is a scalar, the 
%    substraction of the constanct D and Y is computed.
%
%    Examples of usage:
%      >> g = rtcminus(f, g)
%      >> g = rtcminus(f, 4)
%      >> g = rtcminus(3, g)
%
%    See also RTCPLUS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

sa = size(a);
sb = size(b);

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    result = CurveMath.sub(a,b);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'double'))
	result = a.clone();
	result.move(0,-b);
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double'))
   	result = b.clone();
	result.scaleY(-1);
	result.move(0,a);
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    if (sa(1,1) == sb(1,1) && sa(1,2) == sb(1,2))
        for i = (1:sa(1,1))
            for j = (1:sa(1,2))
                result(i,j) = CurveMath.sub(a(i,j),b(i,j));
            end
        end
    else
        error('RTCMINUS - the matrix dimensions of the input arguments must be equal.')
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'double'))
    for i = (1:sa(1,1))
        for j = (1:sa(1,2))
            result(i,j) = a(i,j).clone();
            result(i,j).move(0,-b);
        end
    end
elseif (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double'))
    for i = (1:sb(1,1))
        for j = (1:sb(1,2))
            result(i,j) = b(i,j).clone();
            result(i,j) = result(i,j).scale(-1);
            result(i,j).move(0,a);
        end
    end
else
    error('RTCMINUS - minus is not defined for the passed argument list.')
end