function result = rtcpsu(s,p,b,n)
% RTCPSU   Create the upper curve to a periodic resource model.
%    RTCPSU(S,P,B) creates the upper curve to a periodic resource
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
%    See also RTCPSL, RTCCURVE, RTCFS, RTCBDU, RTCBDL, RTCTDMAU, 
%             RTCTDMAL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtcpsu:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin == 3
    n = '';
elseif nargin < 3
    error('RTCPSU - too few arguments.')
elseif nargin > 4
    error('RTCPSU - too many arguments.')
end

if (~strcmp(class(s),'double') || ~strcmp(class(p),'double') ...
        || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
    error('RTCPSL - psl is not defined for the passed argument list.')
end

if round(p) ~= p
    error('RTCPSU(S,P,B) - P must be a natural number');
end

try
    result = CurveFactory.createUpperPeriodicServiceCurve(s,p,b,n);
catch
    if (s < 0)
        error('RTCPSU(S,P,B) - the following condition must be fulfilled: S >= 0');
    elseif (s > p)
        error('RTCPSU(S,P,B) - the following condition must be fulfilled: S <= P');
    elseif (p < 1)
        error('RTCPSU(S,P,B) - the following condition must be fulfilled: P >= 1');
    elseif (b < 0)
        error('RTCPSU(S,P,B) - the following condition must be fulfilled: B >= 0');
    end
end