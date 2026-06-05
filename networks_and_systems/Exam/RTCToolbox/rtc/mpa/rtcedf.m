function [varargout] = rtcedf(varargin)
% RTCEDF   Compute the output of an EDF processing component.
%    [A1',Del1,Buf1,...,An',Deln,Bufn, B']=RTCEDF(A1,E1,D1,...,An,En,Dn, B)
%    computes the output curves of an EDF processing component that
%    processes a set of n discrete event streams with arrival curves A1,A2,
%    ...,An on an EDF resource with service curve B. E1,E2, ...,En denote
%    the worst case and best case execution demands and D1,D2, ...,Dn
%    denote the deadlines associated to the n input streams.
%
%    The curves are represented as sets Ai = [AUi ALi] where AUi and ALi
%    are the upper and lower arrival curves of event stream i. The output
%    arrival curves Ai' have the same data structure as Ai.
%
%    B = [BU BL] are the upper and lower service curves of the component.
%    The output service curve B' has the same data structure as B.
%
%    Ei = [WCEDi BCEDi] are the worst case and best case execution demands
%    of events in stream i. The execution demand can be a scalar: In this
%    case we have WCEDi = BCEDi (worst case execution demand = best case
%    execution demand). Di are the deadlines associated to events in stream
%    i.
%
%    Deli denote the maximum delays which the events of the streams observe
%    between arrival and output. Bufi denote the maximal buffer space for
%    each stream.
%    
%    The component issues a warning if deadlines are not respected (EDF
%    schedulability test).
%
%    An EDF processing component is triggered by the events of an incoming
%    event stream. A fully preemptable task is instantiated at every event
%    arrival to process the incoming event, and active tasks are processed
%    in a greedy fashion with dynamic priorities according to their
%    deadlines )EDF).
%
%    See also RTCPLOTGPC, RTCGSC, RTCDEL, RTCBUF, RTCGPC
%
%    Author(s): L. Thiele
%    Copyright 2004-2007 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

% nargin-1 should be divisible by 3
if (nargin < 4 || rem((nargin-1),3) ~= 0)
    error('RTCEDF - not the right number of arguments.')
end

% n = number of inputs
n = (nargin-1)/3;

a = java_array('ch.ethz.rtc.kernel.Curve[]',n);
wced = zeros(n,1);
bced = zeros(n,1);
dead = zeros(n,1);
for i=1:n
    a(i) = varargin{3*i-2};
    if(isscalar(varargin{3*i-1}))
        wced(i) = varargin{3*i-1}; 
        bced(i) = wced(i);
    else
        wced(i) = varargin{3*i-1}(1);
        bced(i) = varargin{3*i-1}(2);
    end
    dead(i) = varargin{3*i};
end
b = varargin{nargin};


% Execute schedulability test: BL >= SUM(AUi shifted)
DBF = rtctimes(rtcaffine(a(1,1), 1.0, dead(1)), wced(1));
for i=2:n
    DBF = rtcplus(DBF, rtctimes(rtcaffine(a(i,1), 1.0, dead(i)), wced(i)));
end

maxdist = rtcv(DBF, b(2));
if (maxdist > 0)
    warning('MATLAB:rtcedf:notSchedulable', 'EDF schedulability test failed!');
    figure; rtcplot(DBF,'r', b(2), 'g',50); title('DBF (red); Service (green)');
end

% compute the remaining service curves
RBFl = rtctimes(a(1,2), bced(1));
for i=2:n
    RBFl = rtcplus(RBFl, rtctimes(a(i,2), bced(i)));
end
buo = rtcmaxdeconv(rtcminus(b(1), RBFl), 0);

RBFu = rtctimes(a(1,1), wced(1));
for i=2:n
    RBFu = rtcplus(RBFu, rtctimes(a(i,1), wced(i)));
end
blo = rtcmaxconv(rtcminus(b(2), RBFu), 0);
bres = [buo blo];

% compute the resulting arrival curves, delays and buffer spaces
% (simple formulas)
ares = java_array('ch.ethz.rtc.kernel.Curve[]',n);
delay = zeros(n,1);
buffer = zeros(n,1);
for i=1:n
    % Hint:
    % These formula would be tighter, but there is an issue with
    % the unit of bced. It is currently in [cycles/event] but needs
    % to be in [sec] (bced'). For the transformation, the inverse function
    % of beta would be needed: bced' = inv-beta(bced)
    %
    % auo = rtcaffine(a(i,1), 1.0, bced(i) - dead(i));
    % alo = rtcaffine(a(i,2), 1.0, dead(i) - bced(i));
    
    % alternative solution for bced(i)=0 (pessimistic)
    auo = rtcaffine(a(i,1), 1.0, -dead(i));
    alo = rtcaffine(a(i,2), 1.0, dead(i));
    
    ares(i,:) = [auo alo];
    delay(i) = dead(i);
    buffer(i) = auo.y0epsilon();
end

% compute the resulting arrival curves (complex formula)
for i=1:n
    bu = b(1);
    if n==1 
        bl = b(2);
    else
        RBF = rtcminus(RBFu, rtctimes(a(i,1), wced(i)));
        bl = rtcmaxconv(rtcminus(b(2), RBF), 0);
    end
    bu = rtcrdivide(bu, bced(i));
    bl = rtcrdivide(bl, wced(i));
    
    auo = rtcceil(rtcmin(rtcmindeconv(rtcminconv(a(i,1), bu), bl), bu));
    alo = rtcfloor(rtcmin(rtcminconv(rtcmindeconv(a(i,2), bu), bl), bl));
    ares(i,1) = rtcmin(ares(i,1),auo);
    ares(i,2) = rtcmax(ares(i,2),alo);
    bl = rtcmax(0,rtcfloor(bl));
    delay(i) = min(delay(i), rtch(a(i,1), bl));
    buffer(i) = min(buffer(i), rtcv(a(i,1), bl));
end

for i=1:n
    varargout{3*i-2} = ares(i,:);
    varargout{3*i-1} = delay(i);
    varargout{3*i} = buffer(i);
end
varargout{3*n+1} = bres;
