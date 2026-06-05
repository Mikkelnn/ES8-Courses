function value = rtcvalue( curve, arg, dir )
% RTCVALUE determines the value of a curve at some point
%    Call:   value = rtcvalue( curve, arg, dir )
%    Input:  curve: the curve we are investigating
%            arg: the point at which we determine curve
%            dir >= 0 : the result is curve(arg + epsilon)
%            dir < 0 : the result is curve(arg - epsilon)
%    Output: value of curve at arg
%
%    The function determines the left and right limit value
%    of a curve at some point

import ch.ethz.rtc.kernel.*;

if (dir >= 0) % determine curve(arg + epsilon)
   value = curve.segmentsLEQ(arg).lastSegment().yAt(arg);
else % determine curve(arg - epsilon)
   value = curve.segmentsLT(arg).lastSegment().yAt(arg);
end





