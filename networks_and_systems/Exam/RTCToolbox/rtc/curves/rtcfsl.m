function result = rtcfsl(b, n)
% RTCFSL   Create the lower curve to a full service resource.
%    RTCFSL(B) creates the lower curve to a full service
%    resource (a resource that is fully available) with a 
%    bandwidth B.
%
%    The parameter B can be followed by a character 
%    string N that specifies the optional name of the curve.
%
%    The parameters must fulfill the following conditions:
%      - B >= 0
%
%    See also RTCFSU, RTCCURVE, RTCBDU, RTCBDL, RTCPSU, 
%             RTCPSL, RTCTDMAU, RTCTDMAL, RTCPJDU, RTCPJDL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtcfsl:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin == 1
    n = '';
elseif nargin < 1
    error('RTCFSL - too few arguments.')
elseif nargin > 2
    error('RTCFSL - too many arguments.')
end

if (~strcmp(class(b),'double') || ~strcmp(class(n),'char'))
    error('RTCFSL - fs is not defined for the passed argument list.')
end

if (b < 0)
    error('RTCFSL(B) - the following condition must fulfilled: B >= 0');
end

result = Curve([[0 0 b]],n);