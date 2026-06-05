function d = rtcdel(alpha, varargin)
% RTCDEL   Compute the maximum delay at a GPC.
%    RTCDEL(A,B) computes the maximum delay experienced
%    by an event on a discrete event stream with upper arrival 
%    curve A that is processed by a greedy processing
%    component (RTCGPC) that runs on a resource with 
%    lower service curve B. With these parameters, it is
%    assumed that B is already adapted to the event worst-
%    case execution demand.
%
%    RTCDEL(A,B,WCED) computes the maximum delay experienced
%    by a discrete event with worst-case exeuction demand WCED, 
%    that arrives on an event stream with upper arrival 
%    curve A and that is processed by a greedy processing
%    component (RTCGPC) that runs on a resource with 
%    lower service curve B.
%
%    RTCDEL(A,B1,B2,...,BN) computes the total 
%    end-to-end delay experienced by a event on a discrete event 
%    stream with upper arrival curve A that is processed on
%    N consecutive GPC's with lower serivce curves 
%    B1,...,BN. With these parameters, it is assumed that 
%    the B's are already adapted to the various event worst-
%    case execution demands.
%
%    RTCDEL(A,B1,WCED1,B2,WCED2,...,BN,WCEDN) computes the total 
%    end-to-end delay experienced by an event on a discrete event 
%    stream with upper arrival curve A that is processed on
%    N consecutive GPC's with lower serivce curves 
%    B1,...,BN and with event worst-case execution demands
%    WCED1,...,WCEDN.
%
%    The curves A and Bi can also be replaced by curve sets.
%
%    See also RTCBUF, RTCGPC, RTCH.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

betas = [];
wceds = [];
state  = 0;

for i = (1:length(varargin))
    switch class(varargin{i})
        case 'ch.ethz.rtc.kernel.Curve'
            switch state
                case 0
                    betas = [betas varargin{i}];
                    state = 1;
                case 1
                    betas = [betas varargin{i}];
                    state = 2;
                case 2
                    betas = [betas varargin{i}];
                    state = 2;
                case 3
                    betas = [betas varargin{i}];
                    state = 4;
                otherwise
                    error('RTCDEL - del is not defined for the passed argument list.')
            end
        case 'ch.ethz.rtc.kernel.Curve[]'
            switch state
                case 0
                    betas = [betas varargin{i}(2)];
                    state = 1;
                case 1
                    betas = [betas varargin{i}(2)];
                    state = 2;
                case 2
                    betas = [betas varargin{i}(2)];
                    state = 2;
                case 3
                    betas = [betas varargin{i}(2)];
                    state = 4;
                otherwise
                    error('RTCDEL - del is not defined for the passed argument list.')
            end
        case 'double'
            switch state
                case 1
                    wceds = [wceds varargin{i}];
                    state = 3;
                case 4
                    wceds = [wceds varargin{i}];
                    state = 3;
                otherwise
                    error('RTCDEL - del is not defined for the passed argument list.')
            end
    end
end

switch state
    case {1,2}
        for j = (1:length(betas))
            betas(j) = rtcmax(0,rtcfloor(betas(j)));
        end
    case 3
        for j = (1:length(betas))
            betas(j) = rtcmax(0,rtcfloor(rtcrdivide(betas(j), wceds(j))));
        end        
    otherwise
        error('RTCDEL - del is not defined for the passed argument list.')
end

beta = betas(1);
if length(betas) > 1
    for j = (2:length(betas))
        beta = rtcminconv(beta, betas(j));
    end
end

if (strcmp(class(alpha),'ch.ethz.rtc.kernel.Curve[]'))
    alpha = alpha(1);
end

d = rtch(alpha, beta);