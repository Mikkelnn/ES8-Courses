function ago = rtcspsag(p, e, sigmaai);
% RTCDPSAG 

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

import ch.ethz.rtc.kernel.*;

if nargin < 3
    error('RTCSPSAG - too few arguments.')
elseif nargin > 3
    error('RTCSPSAG - too many arguments.')
end

if (strcmp(class(p),'double') & strcmp(class(e),'double') & strcmp(class(sigmaai),'ch.ethz.rtc.kernel.Curve[]'))
    ago = rtcdpsag(p, e, sigmaai);
else
    error('RTCSPSAG - rtcspsag is not defined for the passed argument list.')
end