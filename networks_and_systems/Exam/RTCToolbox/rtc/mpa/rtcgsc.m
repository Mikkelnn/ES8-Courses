function [auores alores del buf] = rtcgsc(aui, ali, sigma)
% RTCGSC   Compute the output of a greedy shaper component.
%    [AU',AL',Del,Buf] = RTCGSC(AU,AL,SIGMA) computes the output 
%    curves of a greedy shaper component that shapes a discrete event 
%    stream with upper arrival curve AU and lower arrival curve AL 
%    with the shaping curve SIGMA. Del and Buf denote the maximal
%    delay of events and the maximal buffer space requirements.
%
%    A greedy shaper with a shaping curve SIGMA delays events
%    of an input event stream, so that the output event stream
%    has sigma as upper arrival curve, and it outputs all events
%    as soon as possible.
%
%    [A',Del,Buf] = RTCGSC(A,SIGMA) computes the output of a greedy 
%    shaper component that shapes an event stream with arrival curve set A 
%    with the shaping curve SIGMA.
%
%    See also RTCPLOTGSC, RTCGPC, RTCEDF.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

if nargin < 2
    error('GSC - too few arguments.')
elseif nargin == 2
    if (strcmp(class(aui),'ch.ethz.rtc.kernel.Curve[]'))
        sigma = ali;
        ai = aui;
        aui = ai(1);
        ali = ai(2);
    else
    	error('RTCGSC - gsc is not defined for the passed argument list.')
    end
elseif nargin > 3
    error('RTCGSC - too many arguments.')
end

if (strcmp(class(aui),'ch.ethz.rtc.kernel.Curve') && strcmp(class(ali),'ch.ethz.rtc.kernel.Curve') ...
        && strcmp(class(sigma),'ch.ethz.rtc.kernel.Curve'))
    auo = rtcceil(rtcminconv(aui,sigma));
    alo = rtcceil(rtcminconv(ali,rtcmaxdeconv(sigma,sigma))); %% FIXME: is ceil correct here?
else
    error('RTCGSC - gsc is not defined for the passed argument list.')
end

sigma = rtcceil(sigma);
delay =  rtch(aui, sigma);
buffer = rtcv(aui, sigma);

if nargin == 2
    auores = [auo alo];
    alores = delay;
    del = buffer;
else
    auores = auo;
    alores = alo;
    del = delay;
    buf = buffer;
end
