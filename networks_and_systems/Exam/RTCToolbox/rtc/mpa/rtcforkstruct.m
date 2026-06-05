function [A_I, A_II, varargout] = rtcforkstruct(N, M, A, varargin)
% RTCFORKSTRUCT   Computes the decomposition of a structured stream A into two 
%    structured substreams A_I and A_II, including the new Event Count
%    Curves.
%
%    The composition of A is described by N+M Event Count Curves ECC_1, ..., ECC_{N+M}.
%
%    Stream A_I is formed by the the substreams 1 to N of stream A.
%    Stream A_II is formed by the the substreams N+1 to N+M of stream A.
%    
%    [A_I, A_II, ECC_I_1, ..., ECC_I_N, ECC_II_1, ..., ECC_II_M] = RTCFORKSTRUCT(N, M, A, ECC_1, ..., ECC_{N+M}) 
%
%    Author(s): S. Perathoner
%    Copyright 2009 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.CurveMath;

% Check input arguments

if nargin < 5
    error('RTCFORKSTRUCT - Too few arguments.')
end
if ~(isnumeric(N) & isnumeric(M))
   error('RTCFORKSTRUCT - Not defined for the passed argument list.')
end
if nargin ~= (M + N + 3)
    error('RTCFORKSTRUCT - Wrong number of arguments. I need two integers N and M, an Arrival Curve, and N+M Event Count Curves.')
end
for i = 1 : nargin-3
    if (~strcmp(class(varargin{i}),'ch.ethz.rtc.kernel.Curve[]'))
        error('RTCFORKSTRUCT - Not defined for the passed argument list.')
    end
end


% Do the computation

% Compute A_I upper
sum_I = CurveMath.concatUpper(varargin{1}(1), A(1));
for i = 2 : N
    sum_I = rtcplus(sum_I, CurveMath.concatUpper(varargin{i}(1), A(1)));
end
%sum_II = CurveMath.concatUpper(varargin{N+1}(2), A(1));
sum_II = CurveMath.concatLower(varargin{N+1}(2), A(2));
for i = 2 : M
    %sum_II = rtcplus(sum_II, CurveMath.concatUpper(varargin{i+N}(2), A(1)));
    sum_II = rtcplus(sum_II, CurveMath.concatLower(varargin{i+N}(2), A(2)));
end
temp = rtcminus(A(1), sum_II);
%temp = rtcmaxdeconv(temp, 0);
temp = rtcmaxconv(temp, 0);
temp = rtcmin(sum_I, temp);
A_I(1) = temp;



% Compute A_I lower
sum_I = CurveMath.concatLower(varargin{1}(2), A(2));
for i = 2 : N
    sum_I = rtcplus(sum_I, CurveMath.concatLower(varargin{i}(2), A(2)));
end
%sum_II = CurveMath.concatLower(varargin{N+1}(1), A(2));
sum_II = CurveMath.concatUpper(varargin{N+1}(1), A(1));
for i = 2 : M
    %sum_II = rtcplus(sum_II, CurveMath.concatLower(varargin{i+N}(1), A(2)));
    sum_II = rtcplus(sum_II, CurveMath.concatUpper(varargin{i+N}(1), A(1)));
end
temp = rtcminus(A(2), sum_II);
%temp = rtcmaxconv(temp, 0);
temp = rtcmaxdeconv(temp, 0);
temp = rtcmax(sum_I, temp);
A_I(2) = temp;



% Compute A_II upper
sum_II = CurveMath.concatUpper(varargin{N+1}(1), A(1));
for i = 2 : M
    sum_II = rtcplus(sum_II, CurveMath.concatUpper(varargin{i+N}(1), A(1)));
end
%sum_I = CurveMath.concatUpper(varargin{1}(2), A(1));
sum_I = CurveMath.concatLower(varargin{1}(2), A(2));
for i = 2 : N
    %sum_I = rtcplus(sum_I, CurveMath.concatUpper(varargin{i}(2), A(1)));
    sum_I = rtcplus(sum_I, CurveMath.concatLower(varargin{i}(2), A(2)));
end
temp = rtcminus(A(1), sum_I);
%temp = rtcmaxdeconv(temp, 0);
temp = rtcmaxconv(temp, 0);
temp = rtcmin(sum_II, temp);
A_II(1) = temp;


% Compute A_II lower
sum_II = CurveMath.concatLower(varargin{N+1}(2), A(2));
for i = 2 : M
    sum_II = rtcplus(sum_II, CurveMath.concatLower(varargin{i+N}(2), A(2)));
end
%sum_I = CurveMath.concatLower(varargin{1}(1), A(2));
sum_I = CurveMath.concatUpper(varargin{1}(1), A(1));
for i = 2 : N
    %sum_I = rtcplus(sum_I, CurveMath.concatLower(varargin{i}(1), A(2)));
    sum_I = rtcplus(sum_I, CurveMath.concatUpper(varargin{i}(1), A(1)));
end
temp = rtcminus(A(2), sum_I);
%temp = rtcmaxconv(temp, 0);
temp = rtcmaxdeconv(temp, 0);
temp = rtcmax(sum_II, temp);
A_II(2) = temp;



identity = rtcfs(1);

% Compute ECC_I_1 to ECC_I_N upper

for i = 1 : N
    temp = varargin{i}(1);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : N    
       if k ~= i
          sum = rtcplus( sum, CurveMath.concatUpper( varargin{k}(2), temp  ) );
          %sum = rtcplus( sum, CurveMath.concatLower( varargin{k}(2), temp  ) );
       end
    end
    temp = rtcplus(identity(2), sum);
    temp = CurveMath.invert(temp);         
    bound1 = rtcmin(identity(1), temp);
    
    temp = varargin{i}(2);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : M    
        sum = rtcplus( sum, CurveMath.concatLower( varargin{k+N}(1), temp  ) );
        %sum = rtcplus( sum, CurveMath.concatUpper( varargin{k+N}(1), temp  ) );
    end
    temp = varargin{i}(1);
    temp = CurveMath.invert(temp); 
    temp = rtcminus(temp, sum);
    temp = rtcmaxdeconv(temp, 0);    
    %temp = rtcmaxconv(temp, 0);    
    temp = CurveMath.invert(temp);         
    bound2 = rtcmin(identity(1), temp);
          
    varargout{i}(1) = rtcmin(bound1, bound2);    
end



% Compute ECC_I_1 to ECC_I_N lower

for i = 1 : N
    temp = varargin{i}(2);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : N    
       if k ~= i
          sum = rtcplus( sum, CurveMath.concatLower( varargin{k}(1), temp  ) );
          %sum = rtcplus( sum, CurveMath.concatUpper( varargin{k}(1), temp  ) );
       end
    end
    temp = rtcplus(identity(1), sum);
    temp = CurveMath.invert(temp);                
    bound1 = rtcmin(identity(2), temp);

    temp = varargin{i}(1);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : M    
       sum = rtcplus( sum, CurveMath.concatUpper( varargin{k+N}(2), temp  ) );
       %sum = rtcplus( sum, CurveMath.concatLower( varargin{k+N}(2), temp  ) );
    end
    temp = varargin{i}(2);
    temp = CurveMath.invert(temp); 
    temp = rtcminus(temp, sum);
    temp = rtcmaxconv(temp, 0);    
    %temp = rtcmaxdeconv(temp, 0);    
    temp = CurveMath.invert(temp); 
    bound2 = rtcmin(identity(2), temp);
       
    varargout{i}(2) = rtcmax(bound1, bound2);       
end



% Compute ECC_II_1 to ECC_II_M upper

for i = 1 : M
    temp = varargin{i+N}(1);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : M    
       if k ~= i
          sum = rtcplus( sum, CurveMath.concatUpper( varargin{k+N}(2), temp  ) );
          %sum = rtcplus( sum, CurveMath.concatLower( varargin{k+N}(2), temp  ) );
       end
    end
    temp = rtcplus(identity(2), sum);
    temp = CurveMath.invert(temp);         
    bound1 = rtcmin(identity(1), temp);

    temp = varargin{i+N}(2);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : N    
       sum = rtcplus( sum, CurveMath.concatLower( varargin{k}(1), temp  ) );       
       %sum = rtcplus( sum, CurveMath.concatUpper( varargin{k}(1), temp  ) );
    end
    temp = varargin{i+N}(1);
    temp = CurveMath.invert(temp); 
    temp = rtcminus(temp, sum);
    temp = rtcmaxdeconv(temp, 0);    
    %temp = rtcmaxconv(temp, 0);    
    temp = CurveMath.invert(temp); 
    bound2 = rtcmin(identity(1), temp);
    
    varargout{i+N}(1) = rtcmin(bound1, bound2);      
end



% Compute ECC_II_1 to ECC_II_M lower

for i = 1 : M
    temp = varargin{i+N}(2);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : M    
       if k ~= i
          sum = rtcplus( sum, CurveMath.concatLower( varargin{k+N}(1), temp  ) );
          %sum = rtcplus( sum, CurveMath.concatUpper( varargin{k+N}(1), temp  ) );
       end
    end
    temp = rtcplus(identity(1), sum);
    temp = CurveMath.invert(temp);                 
    bound1 = rtcmin(identity(2), temp);
    
    temp = varargin{i+N}(1);
    temp = CurveMath.invert(temp); 
    sum = rtcfsl(0);
    for k = 1 : N    
       sum = rtcplus( sum, CurveMath.concatUpper( varargin{k}(2), temp  ) );
       %sum = rtcplus( sum, CurveMath.concatLower( varargin{k}(2), temp  ) );
    end
    temp = varargin{i+N}(2);
    temp = CurveMath.invert(temp); 
    temp = rtcminus(temp, sum);
    temp = rtcmaxconv(temp, 0);    
    %temp = rtcmaxdeconv(temp, 0);    
    temp = CurveMath.invert(temp); 
    bound2 = rtcmin(identity(2), temp);
       
    varargout{i+N}(2) = rtcmax(bound1, bound2);
end


