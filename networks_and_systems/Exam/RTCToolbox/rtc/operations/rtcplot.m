function rtcplot(varargin)
% RTCPLOT   Plot a curve or curve set.
%    RTCPLOT(C) plots the curve or curve set C for x-values from 0 up 
%    to the maximum x-value of the current figure.
%
%    RTCPLOT(C,X) plots the curve or curve set C for x-values from 0 to X.
%
%    Various line types, plot-symbols and colors may be obtained 
%    with RTCPLOT(C,S) and RTCPLOT(C,X,S) where S is a character 
%    string of the form as it is used in the native Matlab PLOT 
%    function.
%
%    RTCPLOT(C1,C2,...) plots all curves or curve sets C1, C2, ...  
%    for x-values from 0 up to the maximum  x-value of the current figure.
%
%    RTCPLOT(C1,C2,..., X) plots all curves or curve sets C1, C2, ... for
%    x-values from 0 to X.
%
%    RTCPLOT(C1,C2,..., S) and RTCPLOT(C1,C2,..., X, S) plots all 
%    curves or curve sets C1, C2, ... with the line types, plot symbols and 
%    colors as specified in S.
%
%    RTCPLOT(C1,S1,C2,S2,...) and RTCPLOT(C1,S1,C2,S2,..., X) plots all 
%    curves or curve sets C1, C2, ... with the individual line types, plot 
%    symbols and colors as specified in S1, S2, ... .
%
%    The C,S parameter pair, the C,X,S parameter triple, 
%    the C1,C2,...,S parameter list, the C1,C2,...,X,S parameter
%    list, the C1,S1,C2,S2,... parameter list, and the 
%    C1,S1,C2,S2, ..., X parameter list can be followed by 
%    parameter/value pairs to specify additional properties of 
%    the plottet lines, as in the native Matlab PLOT function. 
%
%    Instead of parameter X, RTCPLOT accepts in all versions also a row  
%    vector [XMIN XMAX YMIN YMAX] that sets the scaling for the x- and 
%    y-axes in the plot.
%
%    Examples of usage:
%      >> rtcplot(f)
%      >> rtcplot(f, 30)
%      >> rtcplot(f, 'r', 30)
%      >> rtcplot(f, 'r.-', 30)
%      >> rtcplot(f, g)
%      >> rtcplot(f, g, 30)
%      >> rtcplot(f, g, 30, 'r.-')
%      >> rtcplot(f, 'r', g, 'b', 30)
%      >> rtcplot(f, 'r.-', g, 'bx-', 30)
%      >> rtcplot(f, 'r', g, 'b', 30, 'linewidth', 3)
%      >> rtcplot(f, [10 30 5 25])
%
%    See also graph2d/PLOT, RTCPLOTC, RTCPLOTH, RTCPLOTV, RTCPLOTBOUNDS.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

rtcplotc(varargin{:});