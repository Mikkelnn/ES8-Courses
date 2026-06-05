function b = rtcbuf(alpha,varargin)
% RTCBUF   Compute the buffer requirement at a GPC.
%    RTCBUF(A,B) computes the maximum buffer space that
%    is required to buffer a discrete event stream with upper
%    arrival curve A in the input queue of a greedy
%    processing component (RTCGPC) that runs on a resource
%    with lower service curve B. With these parameters, it is
%    assumed that B is already adapted to the event worst-
%    case execution demand.
%
%    RTCBUF(A,B,WCED) computes the maximum buffer space that
%    is required to buffer a discrete event stream with upper
%    arrival curve A and with event worst-case execution
%    demand WCED in the input queue of a greedy
%    processing component (RTCGPC) that runs on a resource
%    with lower service curve B. 
%
%    RTCBUF(A,B1,B2,...,BN) computes the total maximum 
%    buffer space that is required to buffer a discrete event 
%    stream with arrival curve A that is processed on 
%    N consecutive GPC's with lower serivce curves 
%    B1,...,BN, that use shared memory for their 
%    input queues. With these parameters, it is assumed that 
%    the B's are already adapted to the various event worst-
%    case execution demands.
%
%    RTCBUF(A,B1,WCED1,B2,WCED2,...,BN,WCEDN) computes the total 
%    maximum buffer space that is required to buffer a discrete 
%    event stream with arrival curve A that is processed on 
%    N consecutive GPC's with lower serivce curves 
%    B1,...,BN, that use shared memory for their 
%    input queues. The event worst-case execution demands
%    are WCED1,...,WCEDN.
%
%    The curves A and Bi can also be replaced by curve sets.
%
%    See also RTCDEL, RTCGPC, RTCV.

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
                    error('RTCBUF - buf is not defined for the passed argument list.')
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
                    error('RTCBUF - buf is not defined for the passed argument list.')
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
                    error('RTCBUF - buf is not defined for the passed argument list.')
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
        error('RTCBUF - buf is not defined for the passed argument list.')
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

b = rtcv(alpha, beta);