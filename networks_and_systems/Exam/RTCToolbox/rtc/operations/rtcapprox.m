function result = rtcapprox(c, px0New, pdxNew)
% RTCAPPROX approximates a given set of curves
%   c - set of curves to be approximated
%   px0New - end of the aperiodic part
%   pdxNew - new period for the periodic part (if 0, then only aperiodic)
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

if nargin ~= 3
    error('RTCAPPROX - wrong number of arguments.');
end

if ~strcmp(class(c),'ch.ethz.rtc.kernel.Curve[]')
    error('RTCAPPROX - first argument is not a set of curves.');
end

result = [rtcapproxs(c(1),px0New,pdxNew,1), rtcapproxs(c(2),px0New,pdxNew,0)];
return
