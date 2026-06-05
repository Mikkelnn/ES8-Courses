function result = rtccurve(varargin)
% RTCCURVE   Create a curve and is a wrapper for the Java function Curve.
%    RTCCURVE(AS) creates a curve with an aperiodic part. AS is
%    a matrix with dimension (n,3) that specifies the n segments
%    of the aperiodic part. AS(i,1) and AS(i,2) specify the 
%    x- and y-value of the i-th segment-start-point, and AS(i,3) 
%    its slope.
%    
%    RTCCURVE(PS,P) creates a curve with a periodic part PS and
%    a period P. PS is a matrix with dimension (n,3) that 
%    specifies the n segments of the aperiodic part. PS(i,1) 
%    and PS(i,2) specify the x- and y-value of the i-th segment- 
%    start-point, and AS(i,3) its slope. The periodic part
%    starts at (0,0), and consecutive repetitions are arranged
%    in such a way that no discontinuity occurs between them.
%    
%    RTCCURVE(PS,PY0,P) creates a curve with a periodic part PS and
%    a period P, that starts at a y-coordinate PY0. The periodic part
%    starts at (0,PY0), and consecutive repetitions are arranged
%    in such a way that no discontinuity occurs between them.
%    
%    RTCCURVE(PS,PY0,P,PDY) creates a curve with a periodic part PS and
%    a period P, that starts at a y-coordinate PY0, and that has
%    an offset PDY. The periodic part starts at (0,PY0), and 
%    consecutive repetitions have an offset of PDY on the y-axis.
%    
%    RTCCURVE(AS,PS,PX0,P) creates a curve with an aperiodic part AS
%    and a periodic part PS with period P. The periodic part starts
%    at the x-coordinate PX0 and is arranged on the y-axis in 
%    such a way that no discontinuity occurs between the end of 
%    the aperiodic part and the start of the periodic part. 
%    Consecutive repetitions of the periodic part are arranged
%    in such a way that no discontinuity occurs between them.
%
%    RTCCURVE(AS,PS,PX0,PY0,P) creates a curve with an aperiodic part AS
%    and a periodic part PS with period P. The periodic part starts
%    at (PX0,PY0). Consecutive repetitions of the periodic part are 
%    arranged in such a way that no discontinuity occurs between them.
%
%    RTCCURVE(AS,PS,PX0,PY0,P,PDY) creates a curve with an aperiodic 
%    part AS and a periodic part PS with period P. The periodic part 
%    starts at (PX0,PY0). Consecutive repetitions of the periodic 
%    part have an offset of PDY on the y-axis.
% 
%    All parameter options can be followed by a character string N
%    that specifies the optional name of the curve.
%
%    The function is a wrapper for the Java function Curve and
%    checks the input before passing it unchanged to Curve.
%
%    Examples of usage:
%      >> f = rtccurve([[0 0 1];[2 2 0.25]])
%      >> f = rtccurve([[0 0 1];[2 2 0.25]], 5)
%      >> f = rtccurve([[0 0 1];[2 2 0.25]], 2 , 5)
%      >> f = rtccurve([[0 0 1];[2 2 0.25]], 2, 5, 4)
%      >> f = rtccurve([[0 3 2]], [[0 0 1];[2 2 0.25]], 3, 5)
%      >> f = rtccurve([[0 3 2]], [[0 0 1];[2 2 0.25]], 3, 10, 5)
%      >> f = rtccurve([[0 3 2]], [[0 0 1];[2 2 0.25]], 3, 10, 5, 4)
%

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if (exist('ch.ethz.rtc.kernel.Curve','class') ~= 8) 
	error('MATLAB:rtccurve:KernelNotInstalled',...
		'RTC Toolbox not installed properly!\n>>> Please reinstall the RTC Toolbox');
end

import ch.ethz.rtc.kernel.*;

state     = 0;
stateChar = 0;

for i = (1:length(varargin))
    switch class(varargin{i})
        case 'double'
            switch state
                case 0
                    [dimx dimy] = size(varargin{i});
                    if dimy == 3
                        state = 1;
                    else 
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end                    
                case 1
                    [dimx dimy] = size(varargin{i});
                    if dimy == 3
                        state = 5;
                    elseif isequal(size(varargin{i}), [1 1])
                        state = 2;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end  
                case 2
                    if isequal(size(varargin{i}), [1 1])
                        state = 3;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end 
                case 3
                    if isequal(size(varargin{i}), [1 1])
                        state = 4;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end 
                case 5
                    if isequal(size(varargin{i}), [1 1])
                        state = 6;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end 
                case 6
                    if isequal(size(varargin{i}), [1 1])
                        state = 7;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end     
                case 7
                    if isequal(size(varargin{i}), [1 1])
                        state = 8;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end    
                case 8
                    if isequal(size(varargin{i}), [1 1])
                        state = 9;
                    else
                        error('RTCCURVE - curve is not defined for the passed argument list.');
                    end                        
                otherwise
                    error('RTCCURVE - curve is not defined for the passed argument list.')
            end
        case 'char'
            if stateChar == 0
                stateChar = 1;
            else
                error('RTCCURVE - curve is not defined for the passed argument list.')
            end
        otherwise
            error('RTCCURVE - curve is not defined for the passed argument list.')
    end
end

if state == 1 || state >= 5
    part = varargin{1};
    dim = size(part);
    x = part(1,1);
    if x ~= 0
        error('RTCCURVE - the first segment of the aperiodic part must start at x=0.')
    end
    for i = (2:dim(1))
        if part(i,1) <= x
            error('RTCCURVE - the x-values of the segments of the aperiodic part must be strictly increasing.')
        end
        x = part(i,1);
    end
end

if state == 2 || state == 3 || state == 4
    part = varargin{1};
    dim = size(part);
    x = part(1,1);
    if x ~= 0
        error('RTCCURVE - the first segment of the periodic part must start at x=0 in the relative coordinate system.')
    end
    for i = (2:dim(1))
        if part(i,1) <= x
            error('RTCCURVE - the x-values of the segments of the periodic part must be strictly increasing.')
        end
        x = part(i,1);
    end
end

if state >= 5
    part = varargin{2};
    dim = size(part);
    x = part(1,1);
    if x ~= 0
        error('RTCCURVE - the first segment of the periodic part must start at x=0 in the relative coordinate system.')
    end
    for i = (2:dim(1))
        if part(i,1) <= x
            error('RTCCURVE - the x-values of the segments of the periodic part must be strictly increasing.')
        end
        x = part(i,1);
    end
end

result = Curve(varargin{:});