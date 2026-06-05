function result = rtcpsl(s,p,b,n)
% RTCPSL   Create the lower curve to a periodic resource model.
%    RTCPSL(S,P,B) creates the lower curve to a periodic resource
%    where a share of S is allocated within every period P, on 
%    a resource with total bandwidth B.
%
%    In research literature, the following name substitutions
%    and settings are sometimes used:
%        S = Theta, P = Pi, B = 1
%
%    The parameter triple can be followed by a character 
%    string N that specifies the optional name of the curve.
%
%    The parameters must fulfill the following conditions:
%      - S >= 0 ^ S <= P
%      - P must be a natural number
%      - P >= 1
%      - B >= 0
%
%    See also RTCPSU, RTCCURVE, RTCFS, RTCBDU, RTCBDL, 
%             RTCTDMAU, RTCTDMAL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: psl.m 303 2006-01-09 17:26:41Z wandeler $

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtcpsl:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin == 3
    n = '';
elseif nargin < 3
    error('RTCPSL - too few arguments.')
elseif nargin > 4
    error('RTCPSL - too many arguments.')
end

if (~strcmp(class(s),'double') || ~strcmp(class(p),'double') ...
        || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
    error('RTCPSL - psl is not defined for the passed argument list.')
end

if round(p) ~= p
    error('RTCPSL(S,P,B) - P must be a natural number');
end

try
    result = CurveFactory.createLowerPeriodicServiceCurve(s,p,b,n);
catch
    if (s < 0)
        error('RTCPSL(S,P,B) - the following condition must be fulfilled: S >= 0');
    elseif (s > p)
        error('RTCPSL(S,P,B) - the following condition must be fulfilled: S <= P');
    elseif (p < 1)
        error('RTCPSL(S,P,B) - the following condition must be fulfilled: P >= 1');
    elseif (b < 0)
        error('RTCPSL(S,P,B) - the following condition must be fulfilled: B >= 0');
    end
end