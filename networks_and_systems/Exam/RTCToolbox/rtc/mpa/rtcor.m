function [aures alres] = rtcor(au1, al1, au2, al2)
% RTCOR   Compute the input to a component with OR activation.
%    [AU,AL] = RTCOR(AU1,AL1,AU2,AL2) computes the input curves AU and AL 
%    to a component with OR activation.
%
%    [A] = RTCOR(A1,A2) computes the input curve set A  to a component with 
%    OR activation.
%
%    See also RTCAND.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if nargin < 2
    error('RTCOR - too few arguments.')
elseif nargin == 2
    if (strcmp(class(au1),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(al1),'ch.ethz.rtc.kernel.Curve[]'))
        a1 = au1;
        a2 = al1;
        au1 = a1(1);
        al1 = a1(2);
        au2 = a2(1);
        al2 = a2(2);
    else
    	error('RTCOR - is not defined for the passed argument list.')
    end
elseif nargin == 3
	error('RTCOR - too few arguments.')
elseif nargin > 4
	error('RTCOR - too many arguments.')
end

if (strcmp(class(au1),'ch.ethz.rtc.kernel.Curve') && strcmp(class(al1),'ch.ethz.rtc.kernel.Curve') ...
        && strcmp(class(au2),'ch.ethz.rtc.kernel.Curve') && strcmp(class(al2),'ch.ethz.rtc.kernel.Curve'))
    au = rtcplus(au1, au2);
    al = rtcplus(al1, al2);
else
    error('RTCOR - is not defined for the passed argument list.')
end

if nargin == 2
    aures = [au al];
else
    aures = au;
    alres = al;
end

