function result = rtcbdbsl(d, b, mb)
% RTCBDBSL   Creates a bounded delay lower curve with bounded slope.
%    RTCBDBSL(D, BD, BS) creates the lower curve of a bounded delay
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
%      >> lower = rtcbdbsl(20, 1.3, 2.1)
%
%    See also RTCCURVE, RTCFS, RTCPS, RTCTDMA, RTCPJD, RTCBD, RTCBDBS

import ch.ethz.rtc.kernel.*;

if nargin ~= 3
    error('RTCBDBSL - needs 3 arguments.')
end
if (d < 0)
    error('RTCBDBSL - delay must be non-negative.')
end
if (b < 0)
    error('RTCBDBSL - bandwidth must be non-negative.')
end
if (mb < b)
    error('RTCBDBSL - maximal bandwidth must not be smaller than long term bandwidth.')
end

if (mb == b) || (d * b == 0)
    result = Curve([[0, 0, b]];
else
    burst = d * b / (mb - b);
    result = Curve([[0, 0, 0]; [d, 0, b]]);
end

end