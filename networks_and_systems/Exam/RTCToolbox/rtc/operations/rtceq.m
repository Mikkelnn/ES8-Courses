function result = rtceq(a,b)
% RTCEQ   Equality check of two curves or curve sets.
%    EQ(X,Y) if X and Y are equal, the result is 1. 
%    Otherwise 0. The function is only defined if
%    X and Y are either both curves or both curve sets.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve')) && (strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    if (a.equals(b))
        result = 1;
    else
        aTemp = rtcuplus(a);
        bTemp = rtcuplus(b);
        aTemp.simplify();
        bTemp.simplify();
        if (aTemp.equals(bTemp))
            result = 1;
        else
            result = 0;
        end
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]')) && (strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    result = 1;
    sa = size(a);
    sb = size(b);
    if (sa(1,1) == sb(1,1) && sa(1,2) == sb(1,2))
        for i = (1:sa(1,1))
            for j = (1:sa(1,2))
                result(i,j) = min(result, (rtceq(a(i,j), b(i,j))));
            end
        end
    else
        result = 0;
    end
else
    error('RTCEQ - eq not defined for the passed argument list.')
end