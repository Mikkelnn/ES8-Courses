function [varargout] = rtcfork(varargin)
% RTCFORK   Computes the decomposition A into its substreams using
%    the Event Count Curves.
%    
%    [A1, A2, ...] = RTCFORK(A, ECC1, ECC2, ...)
%
%    Author(s): T. Rein
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
a = varargin{1};

for i=2:nargin
    varargout{i-1} = [...
        CurveMath.concatUpper(varargin{i}(1), a(1)),...
        CurveMath.concatLower(varargin{i}(2), a(2))];    
end

