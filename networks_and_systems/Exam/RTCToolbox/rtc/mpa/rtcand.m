function [aures alres] = rtcand(au1, al1, au2, al2, buf1, buf2)
% RTCAND   Compute the output of a component with AND activation.
%    [AU,AL] = RTCAND(AU1,AL1,AU2,AL2,BUF1,BUF2) computes the input curves 
%    AU and AL to a component with AND activation. There are BUF1 and
%    BUF2 initial tokens in the input buffers.
%
%    [AU,AL] = RTCAND(AU1,AL1,AU2,AL2) computes the input curves 
%    AU and AL to a component with AND activation. There are no initial 
%    tokens in the input buffers. 
%
%    [A] = RTCAND(A1,A2,BUF1,BUF2) computes the input curve set A  to a 
%    component with AND activation. There are BUF1 and BUF2 initial tokens 
%    in the input buffers.
%
%    [A] = RTCAND(A1,A2) computes the input curve set A  to a component
%    with AND activation. There are no initial tokens in the input buffers. 
%
%    See also RTCOR.

%    Author(s): E. Wandeler, L. Thiele
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if nargin < 2
    error('RTCAND - too few arguments.')
end

if (strcmp(class(au1),'ch.ethz.rtc.kernel.Curve[]'))
    if nargin == 2
        buffer1 = 0; buffer2 = 0;
    elseif nargin == 4
        buffer1 = au2; buffer2 = al2;
    else
        error('RTCAND - is not defined for the passed argument list.')
    end
    isSet = true;
    a1 = au1;
    a2 = al1;
    au1 = a1(1);
    al1 = a1(2);
    au2 = a2(1);
    al2 = a2(2);
elseif (strcmp(class(au1),'ch.ethz.rtc.kernel.Curve'))
    isSet = false;
    if nargin == 4
        buffer1 = 0; buffer2 = 0;
    elseif nargin == 6
        buffer1 = buf1; buffer2 = buf2;
    else
        error('RTCAND - is not defined for the passed argument list.')
    end
else
    error('RTCAND - is not defined for the passed argument list.')
end

bd = buffer1 - buffer2;
if min(buffer1, buffer2) ~= 0
    warning('RTCAND - one of the input buffers should be empty');
end

if (strcmp(class(au1),'ch.ethz.rtc.kernel.Curve') && strcmp(class(al1),'ch.ethz.rtc.kernel.Curve') ...
        && strcmp(class(au2),'ch.ethz.rtc.kernel.Curve') && strcmp(class(al2),'ch.ethz.rtc.kernel.Curve'))
    tmp1 = rtcmindeconv(au1,al2);
    tmp1.move(0,bd);
    tmp2 = rtcmindeconv(au2,al1);
    tmp2.move(0,-bd);
    au = rtcmax(rtcmin(tmp1, au2), rtcmin(tmp2, au1));
    tmp1 = rtcmaxdeconv(al1,au2);
    tmp1.move(0,bd);
    tmp2 = rtcmaxdeconv(al2,au1);
    tmp2.move(0,-bd);
    al = rtcmin(rtcmax(tmp2, al2), rtcmax(tmp1, al1));
%    al = rtcmin(al1,al2);
else
    error('RTCAND - is not defined for the passed argument list.')
end

if isSet
    aures = [au al];
else
    aures = au;
    alres = al;
end
