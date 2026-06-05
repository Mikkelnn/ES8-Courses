function result = rtcaffine(c,a,b)
% RTCAFFINE   An affine argument transformation of a curve or curve set.
%    F = RTCAFFINE(C,A,B) if C is a curve or curve set, B is a scalar, and
%    A is a positive scalar, then F(x) = C(1/A (x - B))
%    (or: C(x) = F(A * X + B)) is computed. 
%    Note, if C is a curve with a periodic part, then the period of the
%    periodic part times A must be an integer.
%
%    Examples of usage:
%      >> g = rtcaffine(f, 2, -3)

%    Author(s): E. Wandeler/L. Thiele
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (a <= 0)
    error('RTCAFFINE - Argument A must be a positive scalar.')
end

if (strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(a),'double') && strcmp(class(b),'double'))
    s = size(a);
    result = java_array('ch.ethz.rtc.kernel.Curve[]',s);
    for i = (1:s(1,1))
        for j = (1:s(1,2))
            result(i,j) = c(i,j).clone();
            try
                result(i,j).scaleX(a);
            catch
                error('RTCAFFINE - A * Period of periodic part must be an integer.');
            end
            result(i,j).move(b,0);
        end
    end     
elseif (strcmp(class(c),'ch.ethz.rtc.kernel.Curve') && strcmp(class(a),'double') && strcmp(class(b),'double'))
    result = c.clone();
    try
        result.scaleX(a);
    catch
        error('RTCAFFINE - A * Period of periodic part must be an integer.');
    end
    result.move(b,0);
else
    error('RTCAFFINE - is not defined for the passed argument list.')
end


