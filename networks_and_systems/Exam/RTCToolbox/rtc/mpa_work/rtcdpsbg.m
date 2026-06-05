function bgo = rtcdpsbg(p, e);
% RTCDPSBG

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

import ch.ethz.rtc.kernel.*;

if nargin < 2
    error('RTCDPSBG - too few arguments.')
elseif nargin > 2
    error('RTCDPSBG - too many arguments.')
end

if (strcmp(class(p),'double') & strcmp(class(e),'double'))
    bguo = rtcpjdu(p, 0, 0);
    bguo = rtctimes(bguo, e);
    bglo = rtcpjdl(p, p, 0);
    bglo = rtctimes(bglo, e);
    bgo = [bguo bglo];
else
    error('RTCDPSBG - rtcdpsbg is not defined for the passed argument list.')
end