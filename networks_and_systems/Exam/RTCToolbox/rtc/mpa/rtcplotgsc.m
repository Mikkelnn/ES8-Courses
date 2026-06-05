function [auores alores del buf] = rtcplotgsc(aui, ali, sigma, x)
% RTCPLOTGSC   Compute and plot the output of a greedy shaper component.
%    [AU',AL',Del,Buf] = RTCPLOTGSC(AU,AL,SIGMA,X) computes and plots the 
%    output curves of a greedy shaper component that shapes a discrete 
%    event stream with upper arrival curve AU and lower arrival curve AL 
%    with the shaping curve SIGMA. Del and Buf denote the maximal
%    delay of events and the maximal buffer space requirements.
%
%    The curves are plotted for x-values from 0 to X. Instead of the
%    parameter X, RTCPLOTGSC accepts also a row vector [XMIN XMAX YMIN YMAX] 
%    that sets the scaling for the x- and y-axes in the plot.
%
%    A greedy shaper with a shaping curve SIGMA delays events
%    of an input event stream, so that the output event stream
%    has sigma as upper arrival curve, and it outputs all events
%    as soon as possible.
%
%    [A',Del,Buf] = RTCPLOTGSC(A,SIGMA,X) computes the output of a greedy 
%    shaper component that shapes an event stream with arrival curve set A 
%    with the shaping curve SIGMA.
%
%    See also RTCGSC, RTCPLOTGPC, RTCDEL, RTCBUF, RTCEDF.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if nargin < 3
    error('RTCPLOTGSC - too few arguments.')
elseif nargin == 3
    if (strcmp(class(aui),'ch.ethz.rtc.kernel.Curve[]'))
        ai = aui;
        x = sigma;
        sigma = ali;
        aui = ai(1);
        ali = ai(2);
    else
    	error('RTCPLOTGSC - plotgsc is not defined for the passed argument list.')
    end
elseif nargin > 4
    error('RTCPLOTGSC - too many arguments.')
end

if (strcmp(class(aui),'ch.ethz.rtc.kernel.Curve') && strcmp(class(ali),'ch.ethz.rtc.kernel.Curve') ...
        && strcmp(class(sigma),'ch.ethz.rtc.kernel.Curve') && (strcmp(class(x),'double') || strcmp(class(x),'double[]')))
    [auo alo delay buffer] = rtcgsc(aui, ali, sigma);

    subplot(2,1,1)
    rtcplot(aui,'r',ali,'r',sigma,'g:',x);
    title('Input');
    subplot(2,1,2)
    rtcplot(auo,'r',alo,'r',sigma,'g:',x);
    title('Output');
else
    error('RTCPLOTGSC - plotgsc is not defined for the passed argument list.')
end

if nargin == 3
    auores = [auo alo];
    alores = delay;
    del = buffer;
else
    auores = auo;
    alores = alo;
    del = delay;
    buf = buffer;
end


