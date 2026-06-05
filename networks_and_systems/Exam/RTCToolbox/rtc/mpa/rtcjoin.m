function [varargout] = rtcjoin(varargin)
% RTCJOIN   Computes the combination of the streams A1, ... , Ai
%    including the Event Count Curves.
%
%    [A, ECC1, ..., ECCi] = RTCJOIN(A1, ..., Ai) computes the OR-
%    combination of the input curves Ai and gives the Event Count Curves
%    related to the input curves.
%
%    Author(s): T. Rein, S. Perathoner
%    Copyright 2008 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.CurveMath;

% Check input arguments
if nargin < 2
    error('RTCOR - too few arguments.')
end
for i=1:nargin
    if (~strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve[]'))
        error('RTCOR - and is not defined for the passed argument list.')
    end
end

% Do the computation

identity = rtcfs(1);
% Alternatively, a step function could be chosen!
% identity = rtcpjd(1);

for i=1:nargin
    % rtcor for total stream A
    if (i==1)
        sum(1) = varargin{i}(1);
        sum(2) = varargin{i}(2);
    else
        sum(1) = rtcplus(varargin{i}(1), sum(1));
        sum(2) = rtcplus(varargin{i}(2), sum(2));
    end
    
    % compute Event Count Curve ECCi
    inverse(1) = CurveMath.invert(varargin{i}(1));
    inverse(2) = CurveMath.invert(varargin{i}(2));

    eps(1) = identity(1);
    eps(2) = identity(2);
    for j=1:nargin
        if (i~=j)
            temp(1) = CurveMath.concatLower(varargin{j}(1), inverse(2));
            temp(2) = CurveMath.concatUpper(varargin{j}(2), inverse(1));
            eps(1) = rtcplus(temp(1), eps(1));
            eps(2) = rtcplus(temp(2), eps(2));
        end
    end
    
    varargout{i+1}(1) = CurveMath.invert(eps(2));
    varargout{i+1}(2) = CurveMath.invert(eps(1));
end

varargout{1} = sum;
