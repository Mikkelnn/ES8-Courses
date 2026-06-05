function [r rc] = rtcmatplot(a, x, header)
% RTCMATPLOT   Performs the plot of a matrix of curves
%    RTCMATPLOT(C,X) plots the curves in a matrix A for 
%       x-values from 0 to X. 
%    RTCMATPLOT(C,X,HEADER) plots the curves in a matrix A for 
%       x-values from 0 to X with title HEADER.
%
%    See also RTCPLOT.
%
%    Author(s): L. THIELE
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

import ch.ethz.rtc.kernel.*;

if ~(nargin == 2 || nargin == 3)
    error('RTCMATPLOT - exepts 2 or 3 arguments.')
end

if nargin == 2
    header = '';
end

if ~(strcmp(class(a(1,1)),'ch.ethz.rtc.kernel.Curve'))
    error('RTCMATPLOT - not defined for the class of inputs provided.')
end

if ~(strcmp(class(x),'double'))
    error('RTCMATPLOT - not defined for the class of inputs provided.')
end

sa = [size(a,1) size(a(1),1)];

counter = 1;
for i = (1:sa(1))
	for j = (1:sa(2))
            subplot(sa(1),sa(2),counter); 
            rtcplot(a(i,j),  x);
            title(strcat(header,'(',int2str(i),', ',int2str(j), ')') )
            counter = counter + 1; 
    end
end

   