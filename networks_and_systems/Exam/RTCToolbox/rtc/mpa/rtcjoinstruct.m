function [varargout] = rtcjoinstruct(varargin)
% RTCJOINSTRUCT   Computes the combination of two structured streams A_I and A_II 
%    including the Event Count Curves for all simple substreams.
%
%    The composition of A_I is described by N Event Count Curves ECC_I_1, ..., ECC_I_N.
%    The composition of A_II is described by M Event Count Curves ECC_II_1, ..., ECC_II_M.
%
%    [A, ECC_1, ..., ECC_{N+M}] = RTCJOINSTRUCT(N, M, A_I, A_II, ECC_I_1, ..., ECC_I_N, ECC_II_1, ..., ECC_II_M) computes 
%    the OR-combination A of the input curves A_I and A_II and determines the Event Count Curves
%    for all simple substreams composing A. 
%
%    Author(s): S. Perathoner
%    Copyright 2009 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.CurveMath;

% Check input arguments

if nargin < 6
    error('RTCJOINSTRUCT - Too few arguments.')
end
if ~(isnumeric(varargin{1}) & isnumeric(varargin{2}))
   error('RTCJOINSTRUCT - Not defined for the passed argument list.')
end
N = varargin{1};
M =  varargin{2};
if nargin ~= (M + N + 4)
    error('RTCJOINSTRUCT - Wrong number of arguments. I need two integers N and M, two Arrival Curves and N+M Event Count Curves.')
end
for i=3:nargin
    if (~strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve[]'))
        error('RTCJOINSTRUCT - Not defined for the passed argument list.')
    end
end


% Do the computation

aI = varargin{3};
aII = varargin{4};

[sum, eccI, eccII] = rtcjoin(aI, aII);
varargout{1} = sum;

% varargout{1} = rtcplus(aI, aII);
% identity = rtcfs(1);
% inverse_aI(1) = CurveMath.invert(aI(1));
% sum = CurveMath.concatUpper( aII(2), inverse_aI(1));
% sum = rtcplus(identity(1), sum);
% eccI(1) = CurveMath.invert(sum);

for i = 1 : N   
    varargout{i+1}(1) = CurveMath.concatUpper(varargin{i+4}(1), eccI(1));
    varargout{i+1}(2) = CurveMath.concatLower(varargin{i+4}(2), eccI(2));   
end
for i = 1 : M
    varargout{i+1+N}(1) = CurveMath.concatUpper(varargin{i+4+N}(1), eccII(1));
    varargout{i+1+N}(2) = CurveMath.concatLower(varargin{i+4+N}(2), eccII(2));   
end









