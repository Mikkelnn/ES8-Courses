function bgo = rtcfpba(dgi, agi, bai);
% RTCFPBA

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

import ch.ethz.rtc.kernel.*;

if nargin < 2
    error('RTCFPBA - too few arguments.')
elseif nargin == 2
    bai = rtccurve([0 0 0]);
elseif nargin > 3
    error('RTCFPBA - too many arguments.')
end

if (strcmp(class(dgi),'double') & strcmp(class(agi),'ch.ethz.rtc.kernel.Curve') & strcmp(class(bai),'ch.ethz.rtc.kernel.Curve'))
    c1 = rtcclone(agi);
    c1.move(dgi,0);
    if (bai == rtccurve([0 0 0]))
        bgo = c1;
    else
        c2 = CurveMath.RTinvBeta(bai, agi);
        bgo = rtcmax(c1, c2);
    end
else
    error('RTCFPBA - rtcfpga is not defined for the passed argument list.')
end