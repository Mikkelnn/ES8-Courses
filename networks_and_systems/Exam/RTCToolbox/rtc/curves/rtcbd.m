function [result, result2] = rtcbd(d, b, n)
% RTCBD   Create the curve set of a bounded delay resource model.
%    RTCBD(D,B) creates the curve set of a bounded delay
%    resource, with a maximum delay D and a bandwidth B.
%
%    In research literature, the following name substitutions
%    are sometimes used:
%        D = DELTA/2, B = alpha
%
%    RTCBD(D,B,N) creates the curve set of a bounded delay
%    resource, with a maximum delay D, a bandwidth B and name N. 
%
%    Alternatively, the created Curve can also be interpreted as an 
%    ordinary two-segment curve.
%
%    RTCBD(C) creates a bounded-delay two-segment curve set that upper/lower
%    bounds the curve set C.
%
%    [D,B] = RTCBD(C) computes the delay D and bandwidth B of the 
%    bounded-delay two-segment curve that is a upper/lower bound the curve 
%    set C.
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%      - C must be wide-sense increasing
%
%    Examples of usage:
%      >> f = rtcbd(4, 1)
%      >> [d, b] = rtcbd(f)
%      >> f = rtcbd(g)
%
%    See also RTCCURVE, RTCFS, RTCPS, RTCTDMA, RTCPJD.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtcbd:KernelNotInstalled',...
    	'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin < 1
    error('RTCBD - too few arguments.')
elseif nargin > 3
    error('RTCBD - too many arguments.')
end

if nargin == 1
    if (~strcmp(class(d),'ch.ethz.rtc.kernel.Curve'))
        error('RTCBD(C) - C must be a curve.')
    end
    if nargout <= 1
        try
            resultu = CurveFactory.createUpperBoundedDelayCurve(d(1));
            resultl = CurveFactory.createLowerBoundedDelayCurve(d(2));
            result = [resultu resultl];
        catch
            error('RTCBD(C) - C must be wide-sense increasing');
        end
    elseif nargout == 2
        try
        c = CurveFactory.createUpperBoundedDelayCurve(d(1));
        catch
            error('RTCBD(C) - C must be wide-sense increasing');
        end
        cAper = c.aperiodicPart();
        cSeg  = cAper.segments();
        seg = cSeg.segment(0);
        if seg.s() == 0
            result2u = 0;
            if seg.y() == 0
                resultu = 0;
            else
                resultu = Inf;
            end
        else
            result2u = seg.s();
            resultu = seg.y() ./ seg.s();
        end
        try
        c = CurveFactory.createLowerBoundedDelayCurve(d);
        catch
            error('RTCBD(C) - C must be wide-sense increasing');
        end
        cAper = c.aperiodicPart();
        cSeg  = cAper.segments();
        if cSeg.size() == 1
            % seg = cSeg.segment(0);
            resultl = 0;
            % result2l = seg.s();
        else
            seg = cSeg.segment(1);
            resultl = seg.x();
            % result2l = seg.s();
        end
        result = [resultu resultl];
        result2 = [result2u resultl];
    end
    return
end

if nargin == 2
    n = '';
end

if (~strcmp(class(d),'double') || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
	error('RTCBD - bd is not defined for the passed argument list.')
end

try
    resultu = CurveFactory.createUpperBoundedDelayCurve(d,b,n);
    resultl = CurveFactory.createLowerBoundedDelayCurve(d,b,n);
    result = [resultu resultl];
catch
    if (d < 0)
        error('RTCBD(D,B) - the following condition must be fulfilled: D >= 0');
    elseif (b < 0)
        error('RTCBD(D,B) - the following condition must be fulfilled: B >= 0');
    end
end