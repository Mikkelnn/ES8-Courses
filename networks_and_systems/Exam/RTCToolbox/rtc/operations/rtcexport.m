function result = rtcexport(c)
% RTCEXPORT   Exports a curve or a curve set to a textual representation.
%    RTCEXPORT(C) exports curve or curve set C to a representation  
%    that can directly be used in Matlab to create a new curve with the 
%    same data. 
%
%    This representation can also be read by the Real-Time Calculus
%    Library and can be used to store curves.
%
%    See also RTCCURVE.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]')
    resultu = strcat('rtc', char(c(1).toMatlabString()));
    resultl = strcat('rtc', char(c(2).toMatlabString()));
    result = strcat('[', resultu, ',', resultl, ']');
elseif strcmp(class(c),'ch.ethz.rtc.kernel.Curve')
    result = strcat('rtc', char(c.toMatlabString()));
else
    error('RTCEXPORT - export not defined for the passed argument list.')
end