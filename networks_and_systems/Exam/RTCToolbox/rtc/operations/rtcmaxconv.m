function result = rtcmaxconv(varargin)
% RTCMAXCONV   Max-plus convolution of curves.
%    RTCMAXCONV(X,Y) computes the max-plus convolution of the curves 
%    X and Y.
%
%    RTCMAXCONV(X,S) computes the max-plus convolution of the curve X and 
%    the constant S.     
%
%    RTCMAXCONV(X,0) computes the max-plus convolution of the curve X and 
%    zero. For this computation, a special algorithm is used that is more 
%    efficient.
%
%    RTCMAXCONV(X1,X2,X3,...) computes the max-plus convolution of all X's
%    where the X's may be any number and combination  of curves and 
%    constants.
%
%    See also RTCMAXDECONV, RTCMINCONV, RTCMINDECONV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: maxconv.m 301 2006-01-06 16:58:43Z wandeler $

import ch.ethz.rtc.kernel.*;

if (strcmp(class(varargin{1}),'ch.ethz.rtc.kernel.Curve') || strcmp(class(varargin{1}),'double'))
    result = varargin{1};
else 
    error('RTCMAXCONV - maxconv is not defined for the passed argument list.')
end
for i = (2:length(varargin))
    if (strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
        	result = CurveMath.maxPlusConv(result, varargin{i});
        elseif (strcmp(class(result),'double'))
            if (result == 0)
                result = CurveMath.maxPlusConv0(varargin{i});
            else 
                result = Curve([[0 result 0]]);
                result = CurveMath.maxPlusConv(result, varargin{i});
            end
        end
    elseif (strcmp(class(varargin{i}),'double'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
            if (varargin{i} == 0)
                result = CurveMath.maxPlusConv0(result);
            else 
                varargin{i} = Curve([[0 varargin{i} 0]]);
                result = CurveMath.maxPlusConv(result, varargin{i});
            end
        else 
            if (varargin{i} == 0)
                result = Curve([[0 result 0]]);
                result = CurveMath.maxPlusConv0(result);
            elseif (result == 0)
                varargin{i} = Curve([[0 varargin{i} 0]]);
                result = CurveMath.maxPlusConv0(varargin{i});
            else
                result = Curve([[0 result 0]]);
                varargin{i} = Curve([[0 varargin{i} 0]]);
            	result = CurveMath.maxPlusConv(result, varargin{i});  
            end
        end
    else 
        error('RTCMAXCONV - maxconv is not defined for the passed argument list.')
    end
end