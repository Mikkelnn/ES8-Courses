function r = rtcploth(a,b,varargin)
% RTCPLOTH   Plot maximum horizontal distance b. two curves or curve sets.
%    RTCPLOTH(X,Y) plots the maximum horizontal distance between the 
%    curves or curve sets X and Y.
%
%    H = RTCPLOTH(X,Y) plots the maximum horizontal distance between
%    the curves or curve  sets X and Y, and computes the maximum horizontal 
%    distance H.
%
%    Various line types, plot symbols and colors may be obtained 
%    with RTCPLOTH(X,Y,S) where S is a character string of the form
%    as it is used in the native Matlab PLOT function.
%
%    The X,Y,S parameter triple can be followed by 
%    parameter/value pairs to specify additional properties of 
%    the plotted lines, as in the native Matlab PLOT function. 
% 
%    The parameters must fulfill the following condition:
%      - X and Y must be wide-sense increasing 
%
%    See also RTCH, graph2d/plot, RTCPLOTC, RTCPLOTV, RTCPLOTBOUNDS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: ploth.m 329 2006-02-17 09:57:08Z wandeler $

import ch.ethz.rtc.kernel.*;
import ch.ethz.rtc.kernel.util.Vector;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    try
        result = CurveMath.maxHDistLine(a(1),b(2));
    catch
        error('RTCH(X,Y) - X and Y must be wide-sense increasing curves.')
    end
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    try
        result = CurveMath.maxHDistLine(a,b);
    catch
        error('RTCH(X,Y) - X and Y must be wide-sense increasing curves.')
    end
else
    error('RTCPLOTH - ploth is not defined for the passed argument list.')
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
    if result.xEnd() < xmax;
        plot([result.xStart() result.xEnd()],[result.yStart() result.yEnd()],'g.-','linewidth',2);
    elseif result.xStart() < xmax;
        plot([result.xStart() xmax],[result.yStart() result.yEnd()],'g.-','linewidth',2);
    else 
        warning('PLOTH - maximun horizontal distance lies outside plot range.')
    end  
else 
    if result.xEnd() < xmax;
        plot([result.xStart() result.xEnd()],[result.yStart() result.yEnd()],varargin{:});
    elseif result.xStart() < xmax;
        plot([result.xStart() xmax],[result.yStart() result.yEnd()],varargin{:});
    else 
        warning('PLOTH - maximun horizontal distance lies outside plot range.')
    end  
end
if (nargout == 1) 
    r = result.length();
end

if (~washold)
    hold off;
end
    