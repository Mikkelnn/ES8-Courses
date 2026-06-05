function aao = rtcfpaa(dgi, bgi, bai);
% RTCFPAA

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

import ch.ethz.rtc.kernel.*;

if nargin < 3
    error('RTCFPAA - too few arguments.')
elseif nargin > 3
    error('RTCFPAA - too many arguments.')
end

if (strcmp(class(dgi),'double') & strcmp(class(bgi),'ch.ethz.rtc.kernel.Curve') & strcmp(class(bai),'ch.ethz.rtc.kernel.Curve'))
    c1 = rtcclone(bgi);
    c1.move(-dgi,0);
    if (bai == rtccurve([0 0 0]))
        aao = c1;
    else
        c2 = rtcmaxdeconv(CurveMath.RTinvAlpha(bai, bgi), 0);
        aao = rtcmin(c1, c2);
    end
else
    error('RTCFPBA - rtcfpga is not defined for the passed argument list.')
end