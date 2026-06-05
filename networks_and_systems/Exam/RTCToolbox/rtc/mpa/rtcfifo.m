function [varargout] = rtcfifo(varargin)
% RTCFIFO   Compute the output of an FIFO processing component.
%    [A1',Del1,Buf1,...,An',Deln,Bufn, B']=RTCEDF(A1,E1,...,An,En, B)
%    computes the output curves of a FIFO processing component that
%    processes a set of n discrete event streams with arrival curves A1,A2,
%    ...,An on a FIFO resource with service curve B. E1,E2, ...,En denote
%    the worst case and best case execution demands of the n input streams.
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
%    execution demand).
%
%    Deli denote the maximum delays which the events of the streams observe
%    between arrival and output. Bufi denote the maximal buffer space for
%    each stream.
%
%    A FIFO processing component is triggered by the events of an incoming
%    event stream. A task is instantiated at every event arrival to process
%    the incoming event. Tasks are processed in the order of their arrival
%    times.
%
%    See also RTCPLOTGPC, RTCGSC, RTCDEL, RTCBUF, RTCGPC.
%
%    Author(s): L. Thiele
%    Copyright 2004-2007 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.

% nargin-1 should be divisible by 2
if (nargin < 3 || rem((nargin-1),2) ~= 0)
    error('RTCFIFO - not the right number of arguments.')
end

% n = number of inputs
n = (nargin-1)/2;

a = java_array('ch.ethz.rtc.kernel.Curve[]',n);
wced = zeros(n,1);
for i=1:n
    a(i,:) = varargin{2*i-1};
    if(isscalar(varargin{2*i}))
        wced(i) = varargin{2*i}; 
    else
        wced(i) = varargin{2*i}(1);
    end
end
b = varargin{nargin};

% Determine the minimal 'deadline' d: BL >= SUM(AUi shifted by d)
DBF = rtctimes(a(1,1), wced(1));
for i=2:n
    DBF = rtcplus(DBF, rtctimes(a(i,1), wced(i)));
end
d = rtch(DBF, b(2));

% d has to be finite (schedulability)
if (d == java.lang.Double.POSITIVE_INFINITY)
    figure; rtcplot(DBF,'r', b(2), 'g',50); title('DBF (red); Service (green)');
    error('MATLAB:rtcfifo:notSchedulable','FIFO schedulability test failed!');
end

% perform edf scheduling with the same deadline for each input stream

% prepare the input arguments for the rtcedf(...) method
edfargin = cell(n*3+1,1);
for i=1:n
    edfargin{3*i-2} = varargin{2*i-1}; 
    edfargin{3*i-1} = varargin{2*i};
    edfargin{3*i} = d;   
end
edfargin{3*n+1} = varargin{nargin};

[varargout{1:n*3+1}] = rtcedf(edfargin{:});

