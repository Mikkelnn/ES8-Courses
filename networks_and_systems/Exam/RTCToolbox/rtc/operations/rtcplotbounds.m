function rtcplotbounds(curve, varargin)
% RTCPLOTBOUNDS   Plot the upper and lower bounds to a curve or curve set.
%    RTCPLOTBOUNDS(C) plots the tight upper and lower bounds to 
%    the curve or curve set C.
%
%    Various line types, plot symbols and colors may be obtained 
%    with RTCPLOTBOUNDS(C,S) where S is a character string of the form
%    as it is used in the native Matlab PLOT function.
%
%    The C,S parameter pair can be followed by 
%    parameter/value pairs to specify additional properties of 
%    the plottet lines, as in the native Matlab PLOT function. 
%
%    See also graph2d/PLOT, RTCPLOTC, RTCPLOTH, RTCPLOTV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(curve),'ch.ethz.rtc.kernel.Curve[]'))
    lowerBound = curve(2).tightLowerBound();
    upperBound = curve(1).tightUpperBound();
elseif (strcmp(class(curve),'ch.ethz.rtc.kernel.Curve'))
    lowerBound = curve.tightLowerBound();
    upperBound = curve.tightUpperBound();
else
    error('RTCPLOTBOUNDS - plotbounds is not defined for the passed argument list.')
end

if (ishold)
    washold = 1;
else
    hold on;
    washold = 0;
end

ax = axis();
xmax = ax(2);

if (length(varargin) == 0)
    if lowerBound.x() < xmax;
        plot([upperBound.x() xmax], [lowerBound.y() lowerBound.yAt(xmax)],'g:')
    else 
        warning('RTCPLOTBOUNDS - the lower bound distance lies outside plot range.')
    end  
    if upperBound.x() < xmax;
        plot([upperBound.x() xmax], [upperBound.y() upperBound.yAt(xmax)],'g:');
    else 
        warning('RTCPLOTBOUNDS - the upper bound distance lies outside plot range.')
    end  
else 
    if lowerBound.x() < xmax;
        plot([upperBound.x() xmax], [lowerBound.y() lowerBound.yAt(xmax)], varargin{:})
    else 
        warning('RTCPLOTBOUNDS - the lower bound distance lies outside plot range.')
    end  
    if upperBound.x() < xmax;
        plot([upperBound.x() xmax], [upperBound.y() upperBound.yAt(xmax)], varargin{:});
    else 
        warning('RTCPLOTBOUNDS - the upper bound lies outside plot range.')
    end  
end

if (~washold)
    hold off;
end