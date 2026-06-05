function result = rtcbdbs(d, b, mb)
% RTCBDBS   Creates a bounded delay curve set with bounded slope.
%    RTCBDBS(D, BD, BS) creates the curve set of a bounded delay
%    resource with a maximum delay D, a long term bandwidth b and a
%    maximal bandwidth mb.
%
%    Alternatively, the created curves can also be interpreted as 
%    ordinary two-segment curves.
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%      - MB >= B
%
%    Examples of usage:
%      >> [u, l] = rtcbdbs(20, 1.3, 2.1)
%
%    See also RTCCURVE, RTCFS, RTCPS, RTCTDMA, RTCPJD, RTCBD

import ch.ethz.rtc.kernel.*;

if nargin ~= 3
    error('RTCBDBS - needs 3 arguments.')
end
if (d < 0)
    error('RTCBDBS - delay must be non-negative.')
end
if (b < 0)
    error('RTCBDBS - bandwidth must be non-negative.')
end
if (mb < b)
    error('RTCBDBS - maximal bandwidth must not be smaller than long term bandwidth.')
end

if (mb == b) || (d * b == 0)
    result = [Curve([[0, 0, b]]), Curve([[0, 0, b]])];
else
    burst = d * b / (mb - b);
    result = [Curve([[0, 0, mb]; [burst, burst*mb, b]]), ...
        Curve([[0, 0, 0]; [d, 0, b]])];
end

end