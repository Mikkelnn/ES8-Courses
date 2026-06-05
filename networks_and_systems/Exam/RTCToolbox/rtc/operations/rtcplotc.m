function rtcplotc(varargin)
% RTCPLOTC   Plot a curve or curve set.
%    RTCPLOTC(C) plots the curve or curve set C for x-values from 0 up 
%    to the maximum x-value of the current figure.
%
%    RTCPLOTC(C,X) plots the curve or curve set C for x-values from 0 up 
%    to X.
%
%    Various line types, plot-symbols and colors may be obtained 
%    with RTCPLOTC(C,S) and RTCPLOTC(C,X,S) where S is a character 
%    string of the form as it is used in the native Matlab PLOT 
%    function.
%
%    RTCPLOTC(C1,C2,...) plots all curves or curve sets C1, C2, ...  
%    for x-values from 0 up to the maximum  x-value of the current figure.
%
%    RTCPLOTC(C1,C2,..., X) plots all curves or curve sets C1, C2, ... for
%    x-values from 0 to X.
%
%    RTCPLOTC(C1,C2,..., S) and RTCPLOTC(C1,C2,..., X, S) plots all 
%    curves or curve sets C1, C2, ... with the line types, plot symbols and 
%    colors as specified in S.
%
%    RTCPLOTC(C1,S1,C2,S2,...) and RTCPLOTC(C1,S1,C2,S2,..., X) plots all 
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
%    Instead of parameter X, RTCPLOTC accepts in all versions also a row  
%    vector [XMIN XMAX YMIN YMAX] that sets the scaling for the x- and 
%    y-axes in the plot.
%
%    Examples of usage:
%      >> rtcplotc(f)
%      >> rtcplotc(f, 30)
%      >> rtcplotc(f, 'r', 30)
%      >> rtcplotc(f, 'r.-', 30)
%      >> rtcplotc(f, g)
%      >> rtcplotc(f, g, 30)
%      >> rtcplotc(f, g, 30, 'r.-')
%      >> rtcplotc(f, 'r', g, 'b', 30)
%      >> rtcplotc(f, 'r.-', g, 'bx-', 30)
%      >> rtcplotc(f, 'r', g, 'b', 30, 'linewidth', 3)
%      >> rtcplotc(f, [10 30 5 25])
%
%    See also graph2d/PLOT, RTCPLOT, RTCPLOTH, RTCPLOTV, RTCPLOTBOUNDS.
%
%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.


import ch.ethz.rtc.kernel.*;

curves = [];
styles = {};
xmax   = -1;
state  = 0;
curveset = 0;

for i = (1:length(varargin))
    switch class(varargin{i})
        case 'ch.ethz.rtc.kernel.Curve'
            switch state
                case 0
                    curves = [curves varargin{i}];
                    state = 1;
                    curveset = 0;
                case 1
                    curves = [curves varargin{i}];
                    state = 7;
                    curveset = 0;
                case 4
                    curves = [curves varargin{i}];
                    state = 5;
                    curveset = 0;
                case 7
                    curves = [curves varargin{i}];
                    state = 7;
                    curveset = 0;
                otherwise
                    error('PLOT - plot is not defined for the passed argument list.')
            end
        case 'ch.ethz.rtc.kernel.Curve[]'
            switch state
                case 0
                    curves = [curves varargin{i}(1)];
                    curves = [curves varargin{i}(2)];
                    state = 1;
                    curveset = 1;
                case 1
                    curves = [curves varargin{i}(1)];
                    curves = [curves varargin{i}(2)];
                    state = 7;
                    curveset = 1;
                case 4
                    curves = [curves varargin{i}(1)];
                    curves = [curves varargin{i}(2)];
                    state = 5;
                    curveset = 1;
                case 7
                    curves = [curves varargin{i}(1)];
                    curves = [curves varargin{i}(2)];
                    state = 7;
                    curveset = 1;
                otherwise
                    error('PLOT - plot is not defined for the passed argument list.')
            end
        case 'char'
            switch state
                case 1
                    styles{length(styles) + 1} = varargin{i};
                    if (curveset == 1) 
                        styles{length(styles) + 1} = varargin{i};
                    end
                    state = 4;
                case 2
                    styles{length(styles) + 1} = varargin{i};
                    if (curveset == 1) 
                        styles{length(styles) + 1} = varargin{i};
                    end
                    state = 3;
                    break
                case 5
                    styles{length(styles) + 1} = varargin{i};
                    if (curveset == 1) 
                        styles{length(styles) + 1} = varargin{i};
                    end
                    state = 4;
                case 7
                    styles{length(styles) + 1} = varargin{i};
                    if (curveset == 1) 
                        styles{length(styles) + 1} = varargin{i};
                    end
                    state = 8;
                    break
                case 9
                    styles{length(styles) + 1} = varargin{i};
                    if (curveset == 1) 
                        styles{length(styles) + 1} = varargin{i};
                    end
                    state = 10;
                    break
                case 4
                    i = i - 1;
                    break
                otherwise
                    error('PLOT - plot is not defined for the passed argument list.')
            end
        case 'double'
            switch state
                case 1
                    xmax = varargin{i};
                    state = 2;
                case 4
                    xmax = varargin{i};
                    state = 6;
                    break
                case 7
                    xmax = varargin{i};
                    state = 9;                    
                otherwise
                    error('PLOT - plot is not defined for the passed argument list.')
            end
        otherwise
            error('PLOT - plot is not defined for the passed argument list.')
    end
end

if (ishold)
    washold = 1;
else
    washold = 0;
end

userax = 0;
if (xmax == -1)
    ax = axis();
    xmax = ax(2);
end
if isequal(size(xmax),[1 4])
    userax = 1;
    ax = xmax;
    xmax = ax(1,2);
end

pvlen = length(varargin) - i;

switch state
%     case {1,2}
%         curveData = MatlabUtil.printCurve(curves(1), xmax);
%         plot(curveData(:,1), curveData(:,2));
%     case 3
%         curveData = MatlabUtil.printCurve(curves(1), xmax);
%         if (pvlen == 0)
%             plot(curveData(:,1), curveData(:,2), styles{1});
%         else 
%             plot(curveData(:,1), curveData(:,2), styles{1}, varargin{i+1:pvlen+i});
%         end
    case {1,2,7,9}
        for j = (1:length(curves))
            curveData = MatlabUtil.printCurve(curves(j), xmax);
            plot(curveData(:,1), curveData(:,2));
            hold on
        end  
    case {3,4,6}
        for j = (1:length(curves))
            curveData = MatlabUtil.printCurve(curves(j), xmax);
            if (pvlen == 0)
                plot(curveData(:,1), curveData(:,2), styles{j});
            else
                plot(curveData(:,1), curveData(:,2), styles{j}, varargin{i+1:pvlen+i});
            end
            hold on
        end      
	case {8,10}
        for j = (1:length(curves))
            curveData = MatlabUtil.printCurve(curves(j), xmax);
            if (pvlen == 0)
                plot(curveData(:,1), curveData(:,2), styles{1});
            else 
                plot(curveData(:,1), curveData(:,2), styles{1}, varargin{i+1:pvlen+i});
            end      
            hold on
        end       
end

if userax == 1
    axis(ax)
end

if (~washold)
    hold off;
end