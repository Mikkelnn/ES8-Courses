function result = rtcmindeconv(varargin)
% RTCMINDECONV   Min-plus deconvolution of curves.
%    RTCMINDECONV(X,Y) computes the min-plus deconvolution of the curve X 
%    by Y.
%
%    RTCMINDECONV(X,S) computes the min-plus deconvolution of the curve X 
%    by the constant S.     
%
%    RTCMINDECONV(X1,X2,X3,...) computes the nested min-plus deconvolution 
%    of all X's, where the X's may be any number and combination of curves 
%    and constants. I.e. the result is RTCMINDECONV(RTCMINDECONV(X1,X2),X3)
%
%    NOTE: RTCMINDECONV can potentially return a curve with one aperiodic
%          segment (0, Inf, 0).
%
%    See also RTCMINCONV, RTCMAXCONV, RTCMAXDECONV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: mindeconv.m 301 2006-01-06 16:58:43Z wandeler $

import ch.ethz.rtc.kernel.*;

if (strcmp(class(varargin{1}),'ch.ethz.rtc.kernel.Curve') || strcmp(class(varargin{1}),'double'))
    result = varargin{1};
else 
    error('RTCMINDECONV - mindeconv is not defined for the passed argument list.')
end
for i = (2:length(varargin))
    if (strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
        	result = CurveMath.minPlusDeconv(result, varargin{i});
        elseif (strcmp(class(result),'double'))
                result = Curve([[0 result 0]]);
                result = CurveMath.minPlusDeconv(result, varargin{i});
        end
    elseif (strcmp(class(varargin{i}),'double'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
            varargin{i} = Curve([[0 varargin{i} 0]]);
            result = CurveMath.minPlusDeconv(result, varargin{i});
        else 
            result = Curve([[0 result 0]]);
            varargin{i} = Curve([[0 varargin{i} 0]]);
            result = CurveMath.minPlusDeconv(result, varargin{i});  
        end
    else 
        error('RTCMINDECONV - mindeconv is not defined for the passed argument list.')
    end
end