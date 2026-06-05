function ago = rtcdpsag(p, e, sigmaai);
% RTCDPSAG 

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

import ch.ethz.rtc.kernel.*;

if nargin < 3
    error('RTCDPSAG - too few arguments.')
elseif nargin > 3
    error('RTCDPSAG - too many arguments.')
end

if (strcmp(class(p),'double') & strcmp(class(e),'double') & strcmp(class(sigmaai),'ch.ethz.rtc.kernel.Curve[]'))
    c1 = rtcpjdu(p, 0, 0);
    c1 = rtctimes(c1, e);
    c2 = rtcclone(sigmaai(1));
    c2.move(-p,0);
    aguo = rtcmin(c1, c2);
    c1 = rtcpjdl(p, 0, 0);
    c1 = rtctimes(c1, e);
    aglo = rtcmin(c1, sigmaai(2));
    ago = [aguo aglo];
else
    error('RTCDPSAG - rtcdpsag is not defined for the passed argument list.')
end