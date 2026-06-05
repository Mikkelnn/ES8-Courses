function [result, result2] = rtcbdu(d, b, n)
% RTCBDU   Create the upper curve of a bounded delay resource model.
%    RTCBDU(D,B) creates the upper curve of a bounded delay
%    resource, with a maximum delay D and a bandwidth B.
%
%    In research literature, the following name substitutions
%    are sometimes used:
%        D = DELTA/2, B = alpha
%
%    RTCBDU(D,B,N) creates the upper curve of a bounded delay
%    resource, with a maximum delay D, a bandwidth B and name N. 
%
%    Alternatively, the created Curve can also be interpreted as an 
%    ordinary two-segment curve.
%
%    RTCBDU(C) creates a bounded-delay two-segment curve that is a upper
%    bound to the curve C.
%
%    [D,B] = RTCBDU(C) computes the delay D and bandwidth B of the 
%    bounded-delay two-segment curve that is a upper bound to the curve C.
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%      - C must be wide-sense increasing
%
%    Examples of usage:
%      >> f = rtcbdu(4, 1)
%      >> [d, b] = rtcbdu(f)
%      >> f = rtcbdu(g)
%
%    See also RTCBDL, RTCCURVE, RTCFS, RTCPSU, RTCPSL, RTCTDMAU, 
%             RTCTDMAL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
	error('MATLAB:rtcbdu:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin < 1
    error('RTCBDU - too few arguments.')
elseif nargin > 3
    error('RTCBDU - too many arguments.')
end

if nargin == 1
    if (~strcmp(class(d),'ch.ethz.rtc.kernel.Curve'))
        error('RTCBDU(C) - C must be a curve.')
    end
    if nargout <= 1
        try
            result = CurveFactory.createUpperBoundedDelayCurve(d);
        catch
            error('RTCBDU(C) - C must be wide-sense increasing');
        end
    elseif nargout == 2
        try
        c = CurveFactory.createUpperBoundedDelayCurve(d);
        catch
            error('RTCBDU(C) - C must be wide-sense increasing');
        end
        cAper = c.aperiodicPart();
        cSeg  = cAper.segments();
        seg = cSeg.segment(0);
        if seg.s() == 0
            result2 = 0;
            if seg.y() == 0
                result = 0;
            else
                result = Inf;
            end
        else
            result2 = seg.s();
            result = seg.y() ./ seg.s();
        end
    end
    return
end

if nargin == 2
    n = '';
end

if (~strcmp(class(d),'double') || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
	error('RTCBDU - bdu is not defined for the passed argument list.')
end

try
    result = CurveFactory.createUpperBoundedDelayCurve(d,b,n);
catch
    if (d < 0)
        error('RTCBDU(D,B) - the following condition must be fulfilled: D >= 0');
    elseif (b < 0)
        error('RTCBDU(D,B) - the following condition must be fulfilled: B >= 0');
    end
end