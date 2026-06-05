function [auores alores buores blores del buf] = rtcplotgpc(aui, ali, bui, bli, wced, x)
% RTCPLOTGPC   Compute and plot the output of a greedy processing component.
%    [AU',AL',BU',BL',Del,Buf] = RTCPLOTGPC(AU,AL,BU,BL,ED,X) computes and 
%    plots the output curves of a greedy processing component that processes 
%    a discrete event stream with upper arrival curve AU and lower arrival 
%    curve AL on a resource with upper service curve BU and lower
%    service curve BL. The parameter ED specifies the 
%    execution demand of the events on the input
%    event stream. RTCPLOTGPC returns four curves [AU',AL',BU',BL'],
%    that model the outgoing event stream with the upper 
%    arrival curve AU' and the lower arrival curve AL', and
%    the remaining resources with the upper service curve BU'
%    and the lower service curve BL'.
%    The execution demand can be a scalar: In this case we have
%    WCED = BCED (worst case execution demand = best case execution
%    demand). Otherwise, ED = [WCED BCED].
%
%    The curves are plotted for x-values from 0 to X. Instead of the
%    parameter X, PLOTGPC accepts also a row vector [XMIN XMAX YMIN YMAX] 
%    that sets the scaling for the x- and y-axes in the plot.
%
%    [AU',AL',BU',BL',Del,Buf]] = RTCPLOTGPC(AU,AL,BU,BL,X) computes and 
%    plots the output of a greedy processing component, where the worst-case 
%    execution demand of the events is set to 1.
%
%    [A',B',Del,Buf]] = RTCPLOTGPC(A,B,ED,X) computes and plots the output 
%    of a greedy processing component that processes an event stream with 
%    arrival curve set A, service curve set B and execution demand ED.
%
%    [A',B',Del,Buf]] = RTCPLOTGPC(A,B,X) computes and plots the output of 
%    a greedy processing component that processes an event stream with 
%    arrival curve set A and service curve set B. The execution demand of 
%    the events is set to 1.
%
%    A greedy processing component is triggered by the events of an
%    incoming event stream. A fully preemptable task is instantiated at
%    every event arrival to process the incoming event, and active tasks
%    are processed in a greedy fashion in FIFP order, while being
%    restricted by the availbility of resources.
%
%    See also RTCGPC, RTCPLOTGSC, RTCPLOTDEL, RTCPLOTBUF, RTCEDF

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.


if nargin < 3
    error('RTCPLOTGPC - too few arguments.')
elseif nargin < 5
    if (strcmp(class(aui),'ch.ethz.rtc.kernel.Curve[]') && strcmp(class(ali),'ch.ethz.rtc.kernel.Curve[]'))
        if nargin == 3 % RTCPLOTGPC(A,B,X)
            wced = 1; bced = 1;
            x = bui;
        else % RTCPLOTGPC(A,B,ED,X)
            if(isscalar(bui))
               wced = bui; bced = bui;
            else
               wced = bui(1);
               bced = bui(2);
            end
            x = bli;
        end
        ai = aui;
        bi = ali;
        aui = ai(1);
        ali = ai(2);
        bui = bi(1);
        bli = bi(2);
    else
    	error('RTCPLOTGPC - plotgpc is not defined for the passed argument list.')
    end
elseif nargin == 5 % RTCPLOTGPC(AU,AL,BU,BL,X)
	x = wced;
	wced = 1; bced = 1;
elseif nargin == 6 % RTCPLOTGPC(AU,AL,BU,BL,ED,X)
    if(isscalar(ed))
       wced = ed; bced = ed;
    else
       wced = ed(1);
       bced = ed(2);
    end
elseif nargin > 6
    error('RTCPLOTGPC - too many arguments.')
end

[auo alo buo blo delay buffer] = rtcgpc(aui, ali, bui, bli, [wced bced]);
subplot(2,1,1)
rtcplot(aui,'r',ali,'r',rtcrdivide(bui, bced),'b',rtcrdivide(bli, wced),'b',x);
title('Input');
subplot(2,1,2)
rtcplot(auo,'r',alo,'r',rtcrdivide(buo, bced),'b',rtcrdivide(blo, wced),'b',x);
title('Output');

if nargin < 5
    auores = [auo alo];
    alores = [buo blo];
    buores = delay;
    blores = buffer;
else
    auores = auo;
    alores = alo;    
    buores = buo;
    blores = blo;
    del = delay;
    buf = buffer;
end

