function r = rtcplotv(a,b,varargin)
% RTCPLOTV   Plot the max vertical distance bet. two curves or curve sets.
%    RTCPLOTV(X,Y) plots the maximum vertical distance between the 
%    curves or curve sets X and Y.
%
%    V = RTCPLOTV(X,Y) plots the maximum vertical distance between
%    the curves or curve sets X and Y, and computes the maximum vertical 
%    distance V.
%
%    Various line types, plot symbols and colors may be obtained 
%    with RTCPLOTV(X,Y,S) where S is a character string of the form
%    as it is used in the native Matlab PLOT function.
%
%    The X,Y,S parameter triple can be followed by 
%    parameter/value pairs to specify additional properties of 
%    the plottet lines, as in the native Matlab PLOT function. 
%
%    See also RTCV, graph2d/PLOT, RTCPLOTC, RTCPLOTH, RTCPLOTBOUNDS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;
import ch.ethz.rtc.kernel.util.Vector;

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
    result = CurveMath.maxVDistLine(a(1),b(2));
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve') && strcmp(class(b),'ch.ethz.rtc.kernel.Curve'))
    result = CurveMath.maxVDistLine(a,b);
else
    error('RTCPLOTV - plotv is not defined for the passed argument list.')
end

if (ishold)
    washold = 1;
else
    hold on;
    washold = 0;
end

if (length(varargin) == 0)
    plot([result.xStart() result.xEnd()],[result.yStart() result.yEnd()],'g.-','linewidth',2);
else 
    plot([result.xStart() result.xEnd()],[result.yStart() result.yEnd()],varargin{:});
end
if (nargout == 1) 
    r = result.length();
end

if (~washold)
    hold off;
end