function result = rtctighten(c, n)
% RTCTIGHTEN   Tightens a curve set using additivity.
%    RTCTIGHTEN(X, n) computes strengthened curves from a curve set X 
%    using n fixed point iterations. In case only the upper or 
%    lower curve is required, use X = [upper; rtccurve([[0, 0, 0]])] or 
%    X = [rtccurve([[0, inf, 0]]); lower], respectively.
%

%    Author(s): L. Thiele
%    Copyright 2004-2010 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if ~(strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]') || size(c) ~= 2)
    error('RTCTIGHTEN - needs pair of upper and lower curves.')
end

upper = rtcclone(c(1)); lower = rtcclone(c(2));

if (upper.y0epsilon() == inf)
    for i = (1:n)
        lower = rtcmaxconv(lower, lower);
    end
else
    for i = (1:n)
        upper = rtcmin(rtcminconv(upper, upper), rtcmaxdeconv(upper, lower)); upper.simplify();
        lower = rtcmax(rtcmaxconv(lower, lower), rtcmindeconv(lower, upper)); lower.simplify();
        if ((upper.y0epsilon() < 0) || (lower.y0epsilon() == inf))
            warning('RTCTIGHTEN - most probably is upper < lower.');
            result = c;
            return;
        end
    end
end

result = [upper; lower];

