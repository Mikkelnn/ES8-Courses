function result = rtcmaxdeconv(varargin)
% RTCMAXDECONV   Max-plus deconvolution of curves.
%    RTCMAXDECONV(X,Y) computes the max-plus deconvolution of the 
%    curve X by Y.
%
%    RTCMAXDECONV(X,S) computes the max-plus deconvolution of the
%    curve X by the constant S.     
%
%    RTCMAXDECONV(X,0) computes the max-plus convolution of the curve X 
%    by zero. For this computation, a special algorithm is used that is 
%    more efficient.
%
%    RTCMAXDECONV(X1,X2,X3,...) computes the nested max-plus deconvolution 
%    of all X's, where the X's may be any number and combination of curves 
%    and constants. I.e. the result is RTCMAXDECONV(RTCMAXDECONV(X1,X2),X3)
%
%    See also RTCMAXCONV, RTCMINCONV, RTCMINDECONV.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if (strcmp(class(varargin{1}),'ch.ethz.rtc.kernel.Curve') || strcmp(class(varargin{1}),'double'))
    result = varargin{1};
else 
    error('RTCMAXDECONV - maxdeconv is not defined for the passed argument list.')
end
for i = (2:length(varargin))
    if (strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
        	result = CurveMath.maxPlusDeconv(result, varargin{i});
        elseif (strcmp(class(result),'double'))
                result = Curve([[0 result 0]]);
                result = CurveMath.maxPlusDeconv(result, varargin{i});
        end
    elseif (strcmp(class(varargin{i}),'double'))
        if (strcmp(class(result),'ch.ethz.rtc.kernel.Curve'))
            if (varargin{i} == 0)
                result = CurveMath.maxPlusDeconv0(result);
            else 
                varargin{i} = Curve([[0 varargin{i} 0]]);
                result = CurveMath.maxPlusDeconv(result, varargin{i});
            end
        else 
            if (varargin{i} == 0)
                result = Curve([[0 result 0]]);
                result = CurveMath.maxPlusDeconv0(result);
            else
                result = Curve([[0 result 0]]);
                varargin{i} = Curve([[0 varargin{i} 0]]);
                result = CurveMath.maxPlusDeconv(result, varargin{i});  
            end
        end
    else 
        error('RTCMAXDECONV - maxdeconv is not defined for the passed argument list.')
    end
end