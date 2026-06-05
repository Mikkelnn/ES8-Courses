function [p, j, d, s] = rtcconvpjd(c, varargin)
% RTCCONVPJD determines period, jitter and minimal delay of a set of curves
%   c - set of curves to be approximated
%   p - period of curves (may be provided as an optional second argument)
%   j - jitter of curves
%   d - minimal interarrival distance between two events
%   s - stepsize for a single event (1, if no optional second argument)
%
%   If no second argument is provided, the period is determined
%   automatically using the slope of c and assuming a stepsize of 1.
%   Otherwise, the second argument is the period and the stepsize is
%   determined.
%
%   Preconditions: slopes of upper and lower curves should be equal;
%   1/slope of input curves needs to be an integer if no period is provided;
%   the provided period needs to be a positive integer;
%
%    Author(s): L. Thiele
%    Copyright 2004-2008 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
    rtc_init;
end
import ch.ethz.rtc.kernel.*;

if ((nargin > 2) || (nargin == 0))
    error('RTCCONVPJD - wrong number of arguments.');
end

if ~strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]')
    error('RTCCONVPJD - first argument is not a set of curves.');
end

upper = c(1);
lower = c(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine Period and Stepsize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

slope = upper.tightUpperBound.s;
slope1 = lower.tightUpperBound.s;

if (slope ~= slope1)
    error('RTCCONVPJD - upper and lower curves have different slope.');
end

if (nargin == 1)
    period = round(1/slope);
    step = 1;
    if (abs(1.0 - period*slope) > 10^(-6)) 
        error('RTCCONVPJD - 1/slope of input curves is not an integer.');
    end
else
    period = varargin{1};
    step = slope*period;
end

p = period;
s = step;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine Jitter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

A = rtctimes(rtcpjd(period), step);
jitteru = rtch(upper, A(1));
jitterl = rtch(A(2), lower);
jitter = max(jitteru, jitterl);

j = jitter;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determine Delay
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% We use binary search here; may be that there are better options ... 
% Problem: If d is close to p, then the representation of the curve is
% very long. Therefore, we need to approach d = p slowly.

% determine lower bound
dLower = 0;

% determine upper bound
dUpper = 0.;
diff = period/2;
while (dUpper/period < 0.95) % make sure that we are not too close to period
    A = rtctimes(rtcpjdu(period, jitter, dUpper), step);
    distance = rtch(upper, A);
    if (distance > 0) % not feasible
        break;
    end
    dUpper = dUpper + diff;
    diff = diff/2;
end
if (distance == 0) % d is very close to period; just stop to come closer
    d = dUpper - 2 * diff;
    return
end

% perform binary search; dLower feasible, dUpper not feasible
while ((dUpper - dLower)/period > 0.0001) % precision
    d = (dUpper + dLower)/2;
    A = rtctimes(rtcpjdu(period, jitter, d), step);
    distance = rtch(upper, A);
    if (distance == 0) % feasible
        dLower = d;
    else % not feasible
        dUpper = d;
    end
end
d = dLower;

return
