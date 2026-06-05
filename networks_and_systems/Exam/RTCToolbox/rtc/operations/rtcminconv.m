function result = rtcminconv(varargin)
% RTCMINCONV   Min-plus convolution of curves.
%    RTCMINCONV(X,Y) computes the min-plus convolution of the curves 
%    X and Y.
%
%    RTCMINCONV(X,S) computes the min-plus convolution of the curve X and 
%    the constant S.     
%
%    RTCMINCONV(X1,X2,X3,...) computes the min-plus convolution of all X's 
%    where the X's may be any number and combination of curves and 
%    constants.
%
%    See also RTCMINDECONV, RTCMAXCONV, RTCMAXDECONV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(varargin{1}),'ch.ethz.rtc.kernel.Curve') || strcmp(class(varargin{1}),'double'))
    result = varargin{1};
else 
    error('RTCMINCONV - minconv is not defined for the passed argument list.')
end
for i = (2:length(varargin))
    if (strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
        	result = CurveMath.minPlusConv(result, varargin{i});
        elseif (strcmp(class(result),'double'))
            result = Curve([[0 result 0]]);
            result = CurveMath.minPlusConv(result, varargin{i});
        end
    elseif (strcmp(class(varargin{i}),'double'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
            varargin{i} = Curve([[0 varargin{i} 0]]);
            result = CurveMath.minPlusConv(result, varargin{i});
        else 
            result = Curve([[0 result 0]]);
            varargin{i} = Curve([[0 varargin{i} 0]]);
         	result = CurveMath.minPlusConv(result, varargin{i});            
        end
    else 
        error('RTCMINCONV - minconv is not defined for the passed argument list.')
    end
end