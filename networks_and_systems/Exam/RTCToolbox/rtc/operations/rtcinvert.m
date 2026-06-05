function c = rtcinvert(a)
% RTCINVERT   Inverts a curve or curve set.
%   C = RTCINVERT(A) If A is a single curve, C is the inverse of A. If A
%   is a curve set, then C(1) is the inverse of A(2) and C(2) is the
%   inverse of A(1). Therefore C(1) is an upper and C(2) a lower curve.
%       C(1) = RTCINVERT(A(2))
%       C(2) = RTCINVERT(A(1))
%   
%   The curve(s) must fulfill the following conditions:
%       - The curve(s) must be wide-sense increasing
%       - The period(s) must be integer

%   Author(s): T. Rein
%   Copyright 2008 Computer Engineering and Networks Laboratory (TIK) 
%   ETH Zurich, Switzerland.
%   All rights reserved.

import ch.ethz.rtc.kernel.CurveMath;

% Check input arguments
if nargin ~= 1
    error('RTINVERT - too few/much arguments.')
end

if (strcmp(class(a),'ch.ethz.rtc.kernel.Curve[]'))
    c(2) = CurveMath.invert(a(1));
    c(1) = CurveMath.invert(a(2));
elseif (strcmp(class(a),'ch.ethz.rtc.kernel.Curve'))
    c = CurveMath.invert(a);
else
    error('RTINVERT - and is not defined for the passed argument list.')
end

end
