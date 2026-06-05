function bgo = rtcfpbg(agi, bgi);
% RTCFPBG

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id:$

if nargin < 2
    error('RTCFPBG - too few arguments.')
elseif nargin > 2
    error('RTCFPBG - too many arguments.')
end

if (strcmp(class(agi),'ch.ethz.rtc.kernel.Curve') & strcmp(class(bgi),'ch.ethz.rtc.kernel.Curve'))
    if (agi.upperBound().s() > bgi.lowerBound().s())
        disp('Inifinte Delay');
        bgo = rtccurve([0 0 0]);
    else
        bgo = rtcmaxconv((bgi - agi), 0);
    end
else
    error('RTCFPBG - fpgb is not defined for the passed argument list.')
end