function result = rtcmd(d, b, n)
% RTCMD   Create the curve set of a maximum delay resource model.
%    RTCMD(D,B) creates the curve set of a maximum delay
%    resource, with a maximum delay D and a bandwidth B.
%    The upper and lower curves are:
%
%        Upper(t) = B . t
%        Lower(t) = Max[0, B . (t-D)]
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%    
%    The parameter tuple can be followed by a character 
%    string N that specifies the optional name of the curve.
%
%    Examples of usage:
%      >> f = rtcmd(4, 1)
%
%    See also RTCCURVE, RTCFS, RTCPS, RTCTDMA, RTCPJD, RTCBD.

%    Author(s): L. Thiele
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    error('MATLAB:rtcmd:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end
import ch.ethz.rtc.kernel.*;

if nargin < 2
    error('RTCMD - too few arguments.')
elseif nargin > 3
    error('RTCMD - too many arguments.')
end

if (~strcmp(class(d),'double') || ~strcmp(class(b),'double'))
	error('RTCMD - b or d are not defined for the passed argument list.')
end

if (d < 0)
    error('RTCMD(D,B) - the following condition must be fulfilled: D >= 0');
elseif (b < 0)
    error('RTCMD(D,B) - the following condition must be fulfilled: B >= 0');
end

if nargin == 2
    n = '';
end

resultu = Curve([[0 0 b]],n);
resultl = Curve([[0 0 0]; [d 0 b]],n);
result = [resultu resultl];

end