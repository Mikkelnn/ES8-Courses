function result = rtcapproxs(c, px0New, pdxNew, isUpper)
% RTCAPPROXS approximates a given SINGLE curve
%   c - curve to be approximated
%   px0New - end of the aperiodic part
%   pdxNew - new period for the periodic part (if 0, then only aperiodic)
%   isUpper - 1 if upper curve; 0 if lower curve
% 
%   pdxNew needs to be an integer, pdxNew < Inf. 
%   px0New needs to be < Inf if c has a periodic part, px0New >= 0.
%
%    Author(s): N. Stoimenov, L. Thiele
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

if nargin ~= 4
    error('RTCAPPROXS - wrong number of arguments.');
end

if (pdxNew < 0) || isinf(pdxNew)
    error('RTCAPPROXS - pdxNew out of range.');    
end

if (px0New < 0)
    error('RTCAPPROXS - px0New out of range.');    
end

if ~strcmp(class(c),'ch.ethz.rtc.kernel.Curve')
    error('RTCAPPROXS - first argument is not a curve.');
end

if c.hasPeriodicPart && isinf(px0New)
    error('RTCAPPROXS - px0New out of range.');    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Initial Aperiodic Part
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% determine part of c which is after px0New and its initial y-value
rLast = c.clone();
rLast.move(-px0New, 0);   % rLast: part of curve after px0New
py0New = rLast.y0epsilon; % py0Value: initial y-value
    
% determine initial part of curve rInit
rInit = rtcmin(c, Curve([0, py0New, 0])); % cut at y-value py0New

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Aperiodic Approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (pdxNew == 0)
    if (px0New == Inf)
        result = c;
        result.simplify();
        return
    end
    % determine slope
    if rLast.hasPeriodicPart
        slope = rLast.pdy/rLast.pdx;
    else
        slope = rLast.aperiodicSegments.lastSegment.s;
    end
    % determine position of last segment
    if (isUpper == 1)
        distance = rtcv(rLast, Curve([[0, 0, slope]]));
    else
        distance = py0New - rtcv(Curve([[0, py0New, slope]]), rLast);
    end
    
    if (px0New == 0)
        rTail = Curve([[0, distance, slope]]);
    else
        rTail = Curve([[0,0,0]; [px0New, distance, slope]]);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct Periodic Approximation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (pdxNew > 0)
    % determine slope of periodic part and number of unfoldings
    if rLast.hasPeriodicPart
        slope = rLast.pdy/rLast.pdx;
        xLast = rLast.px0 + lcm(pdxNew, rLast.pdx);
        xNumber = ceil(xLast/pdxNew); % number of unfoldings
    else
        slope = rLast.aperiodicSegments.lastSegment.s;
        xLast = rLast.aperiodicSegments.lastSegment.x;
        xNumber = ceil(xLast/pdxNew) + 1; % number of unfoldings
    end
    
    % Simplify segments that need to be approximated
    tmp = rLast.clone();
    tmp.move(- xNumber * pdxNew , 0);
    yMax = tmp.y0epsilon; % value of rLast at x=xNumber*pdxNew
    rLast = rtcmin(rLast, Curve([0, yMax, 0]));
    
    % approximate last segments
    current = rLast.clone();
    if (isUpper == 1)
        rPer = Curve([0,0,0]);
    else
        rPer = Curve([0,Inf,0]);
    end
    for count=1:xNumber
        next = current.clone();
        next.move(-pdxNew, 0);
        yMax = next.y0epsilon;
        next.move(0, -slope*pdxNew);
        current = rtcmin(current, Curve([0, yMax, 0]));
        if (isUpper == 1)
            rPer = rtcmax(rPer, current); 
        else
            rPer = rtcmin(rPer, current); 
        end
        current = next.clone();
    end
    
    % remove all segments with x values larger or equal to period
    tmp = rPer.aperiodicSegments;
    tmp.trimLT(pdxNew);
    rPerMatlab = eval(tmp.toMatlabString); % convert to matrix representation
    
    % determine periodic part of output
    if (px0New == 0)
        rTail = Curve(rPerMatlab, 0, pdxNew, pdxNew*slope);
    else
        rTail = Curve([[0,0,0]],rPerMatlab, px0New, 0, pdxNew, pdxNew*slope);
    end
end

% combine all parts
result = rtcmax(rTail, rInit);
    
% make sure, that it is an increasing function
if (isUpper == 0)
    result = rtcmaxconv(result, 0);
end
    
result.simplify();
return
