function result = rtcsteps(c)
% RTCSTEPS   Step curves (floor and ceil) of a curve set.
%    RTCSTEPS(X) computes the stepwise curves of the curve set X. I.e. the
%    floor of the lower curve and the ceil of the upper curve is computed.
%
%    See also RTCFLOOR, RTCCEIL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: steps.m 333 2006-02-17 16:59:14Z wandeler $

import ch.ethz.rtc.kernel.*;

result = [CurveMath.ceil(c(1)) CurveMath.floor(c(2))];
