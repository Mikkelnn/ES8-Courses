% Demo: Pay Bursts Only Once.
% In this demo, the Pay Bursts Only Once Problem from the Leiden Workshop 
% 2005 is analyzed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
%
% Problem 9: Pay Bursts Only Once
% ---------------------------------------------------------------------------
% Author  : Ernesto Wandeler, wandeler@tik.ee.ethz.ch
% Date    : 2005-11-24
% Category: small
% ---------------------------------------------------------------------------
% 
% A) Description:
% 
% An event stream with jitter/burst is processed by a row of 3 tasks running
% on independent resources. The end-to-end delay of the events is smaller
% than the sum of the individual delays at the three different tasks. 
% I.e. delay(end-to-end) <= delay(T1) + delay(T2) + delay(T3)
% 
% Assumptions:
% 1) A task can execute as soon as its predecessor has finished.
% 
% B) Contact:
% 
% Ernesto Wandeler, wandeler@tik.ee.ethz.ch
% Lothar Thiele, thiele@tik.ee.ethz.ch
% 
% C) References:

% Author(s): E. Wandeler
% Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
% ETH Zurich, Switzerland.
% All rights reserved.
% $Id: PayBurstsOnlyOnce.m 333 2006-02-17 16:59:14Z wandeler $


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Event stream
% =========================================================================
p = 10;
j = 50;
d = 1;

% Worst-case execution demands
% =========================================================================
wced1 = 1;
wced2 = 4;
wced3 = 8;

% Processor speeds
% =========================================================================
cpu1 = 1;
cpu2 = 1;
cpu3 = 1;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
% Construct the input arrival curves
% =========================================================================
a0 = rtcpjd(p,j,d);

% Construct the service curves
% =========================================================================
b10 = rtcfs(cpu1);
b20 = rtcfs(cpu2);
b30 = rtcfs(cpu3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components.
% =========================================================================
[a1 b11] = rtcgpc(a0, b10, wced1);
[a2 b21] = rtcgpc(a1, b20, wced2);
[a3 b31] = rtcgpc(a2, b30, wced3);

% Compute the total delay through service curve convolution.
% =========================================================================
delay_conv = rtcdel(a0,b10,wced1,b20,wced2,b30,wced3);

tTot = toc;

% Compute the total delay as the sum from the individual delays.
% =========================================================================
delay_add = rtcdel(a0,b10,wced1) + rtcdel(a1,b20,wced2) + rtcdel(a2,b30,wced3);


% Display the results.
% =========================================================================
disp(['Total Delay (computed with convolutions) : ' num2str(delay_conv)])
disp(['Sum of the individual delays             : ' num2str(delay_add)])
disp(['Analysis Time                            : ' num2str(tTot) 's'])
