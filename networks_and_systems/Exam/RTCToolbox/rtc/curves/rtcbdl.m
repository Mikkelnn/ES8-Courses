function [result, result2] = rtcbdl(d, b, n)
% RTCBDL   Create the lower curve of a bounded delay resource model.
%    RTCBDL(D,B) creates the lower curve of a bounded delay
%    resource, with a maximum delay D and a bandwidth B.
%
%    In research literature, the following name substitutions
%    are sometimes used:
%        D = DELTA/2, B = alpha
%
%    RTCBDL(D,B,N) creates the lower curve of a bounded delay
%    resource, with a maximum delay D, a bandwidth B and name N. 
%
%    Alternatively, the created Curve can also be interpreted as an 
%    ordinary two-segment curve.
%
%    RTCBDL(C) creates a bounded-delay two-segment curve that is a lower
%    bound to the curve C.
%
%    [D,B] = RTCBDL(C) computes the delay D and bandwidth B of the 
%    bounded-delay two-segment curve that is a lower bound to the curve C.
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%      - C must be wide-sense increasing
%
%    Examples of usage:
%      >> f = rtcbdl(4, 1)
%      >> [d, b] = rtcbdl(f)
%      >> f = rtcbdl(g)
%
%    See also RTCBDU, RTCCURVE, RTCFS, RTCPSU, RTCPSL, 
%             RTCTDMAU, RTCTDMAL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8)
	error('MATLAB:rtcbdl:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin < 1
    error('RTCBDL - too few arguments.')
elseif nargin > 3
    error('RTCBDL - too many arguments.')
end

if nargin == 1
    if (~strcmp(class(d),'ch.ethz.rtc.kernel.Curve'))
        error('RTCBDL(C) - C must be a curve.')
    end
    if nargout <= 1
        try
            result = CurveFactory.createLowerBoundedDelayCurve(d);
        catch
            error('RTCBDL(C) - C must be wide-sense increasing');
        end
    elseif nargout == 2
        try
        c = CurveFactory.createLowerBoundedDelayCurve(d);
        catch
            error('RTCBDL(C) - C must be wide-sense increasing');
        end
        cAper = c.aperiodicPart();
        cSeg  = cAper.segments();
        if cSeg.size() == 1
            seg = cSeg.segment(0);
            result = 0;
            result2 = seg.s();
        else
            seg = cSeg.segment(1);
            result = seg.x();
            result2 = seg.s();
        end
    end
    return
end

if nargin == 2
    n = '';
end

if (~strcmp(class(d),'double') || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
	error('RTCBDL - bdl is not defined for the passed argument list.')
end

try
    result = CurveFactory.createLowerBoundedDelayCurve(d,b,n);
catch
    if (d < 0)
        error('RTCBDL(D,B) - the following condition must be fulfilled: D >= 0');
    elseif (b < 0)
        error('RTCBDL(D,B) - the following condition must be fulfilled: B >= 0');
    end
end