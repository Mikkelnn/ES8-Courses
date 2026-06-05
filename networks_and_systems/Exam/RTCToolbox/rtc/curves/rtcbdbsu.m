function result = rtcbdbsu(d, b, mb)
% RTCBDBSU   Creates a bounded delay upper curve with bounded slope.
%    RTCBDBSU(D, BD, BS) creates the upper curve of a bounded delay
%    resource with a maximum delay D, a long term bandwidth b and a
%    maximal bandwidth mb.
%
%    Alternatively, the created curve can also be interpreted as an 
%    ordinary two-segment curve.
%
%    The parameters must fulfill the following conditions:
%      - D >= 0
%      - B >= 0
%      - MB >= B
%
%    Examples of usage:
%      >> upper = rtcbdbsu(20, 1.3, 2.1)
%
%    See also RTCCURVE, RTCFS, RTCPS, RTCTDMA, RTCPJD, RTCBD, RTCBDBS

import ch.ethz.rtc.kernel.*;

if nargin ~= 3
    error('RTCBDBSU - needs 3 arguments.')
end
if (d < 0)
    error('RTCBDBSU - delay must be non-negative.')
end
if (b < 0)
    error('RTCBDBSU - bandwidth must be non-negative.')
end
if (mb < b)
    error('RTCBDBSU - maximal bandwidth must not be smaller than long term bandwidth.')
end

if (mb == b) || (d * b == 0)
    result = Curve([[0, 0, b]]);
else
    burst = d * b / (mb - b);
    result = Curve([[0, 0, mb]; [burst, burst*mb, b]]);
end

end