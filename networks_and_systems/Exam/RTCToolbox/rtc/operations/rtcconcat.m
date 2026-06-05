function c = rtcconcat(a, b, uplow)
% RTCCONCAT   Computes the concatenation of curves or curves sets.
%	C = RTCCONCAT(A, B) computes the curve set C as concatenation of
%   the curve sets A and curve set B using
%       C(1) = rtcconcat(A(1), B(1), 'upper')
%       C(2) = rtcconcat(A(2), B(2), 'lower')
%
%   C = RTCCONCAT(A, B, UPLOW) computes the concatenation of curve A and B
%   interpreted as specified in UPLOW. UPLOW is either 'upper' or 'lower'.
%    
%   The curves must fulfill the following conditions:
%       - Both curves must be wide-sense increasing
%       - The period of B must be integer
%       

%	Author(s): T. Rein
%	Copyright 2008 Computer Engineering and Networks Laboratory (TIK) 
%	ETH Zurich, Switzerland.
%	All rights reserved.

import ch.ethz.rtc.kernel.CurveMath;

% Check input arguments and perform operations

if (nargin == 2)
    if (~strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]') || ~strcmp(class(b),'ch.ethz.rtc.kernel.Curve[]'))
        error('RTCCONCAT - is not defined for the passed argument list.')
    end
    c(1) = CurveMath.concatUpper(a(1), b(1));
    c(2) = CurveMath.concatLower(a(2), b(2)); 
elseif (nargin == 3)
    if (~strcmp(class(a),'ch.ethz.rtc.kernel.Curve') || ~strcmp(class(b),'ch.ethz.rtc.kernel.Curve') || ~ischar(uplow))
        error('RTCCONCAT - is not defined for the passed argument list.')
    end
    if (strcmp(uplow, 'upper'))
        c = CurveMath.concatUpper(a, b);
    elseif (strcmp(uplow, 'lower'))
        c = CurveMath.concatLower(a, b);
    else 
        error('RTCCONCAT - the last argument should be either char(upper) or char(lower).')
    end
else
    error('RTCCONCAT - too few/many arguments.')
end
   
end

