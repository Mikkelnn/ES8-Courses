function result = rtctdmau(s, c, b, n)
% RTCTDMAU   Create the upper curve to a TDMA resource model.
%    RTCTDMAU(S,C,B) creates the upper curve to a TDMA resource
%    where a slot of length S is allocated on a resource 
%    with total bandwidth B and TDMA cycle length C.
%
%    The parameter triple can be followed by a character 
%    string N that specifies the optional name of the curve.
%
%    The parameters must fulfill the following conditions:
%      - S >= 0 ^ S <= C
%      - C must be a natural number
%      - C >= 1
%      - B >= 0
%
%    See also RTCTDMAL, RTCCURVE, RTCFS, RTCBDU, RTCBDL, 
%             RTCPSU, RTCPSL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtctdmau:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin == 3
    n = '';
elseif nargin < 3
    error('RTCRTCRTCTDMAU - too few arguments.')
elseif nargin > 4
    error('RTCTDMAU - too many arguments.')
end

if (~strcmp(class(s),'double') || ~strcmp(class(c),'double') ...
        || ~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
    error('RTCTDMAU - tdmau is not defined for the passed argument list.')
end

if round(c) ~= c
    error('RTCTDMAU(S,C,B) - C must be a natural number');
end

try
    result = CurveFactory.createUpperTDMACurve(s,c,b,n);
catch
    if (s < 0)
        error('RTCTDMAU(S,C,B) - the following condition must be fulfilled: S >= 0');
    elseif (s > c)
        error('RTCTDMAU(S,C,B) - the following condition must be fulfilled: S <= C');
    elseif (c < 1)
        error('RTCTDMAU(S,C,B) - the following condition must be fulfilled: C >= 1');
    elseif (b < 0)
        error('RTCTDMAU(S,C,B) - the following condition must be fulfilled: B >= 0');
    end
end