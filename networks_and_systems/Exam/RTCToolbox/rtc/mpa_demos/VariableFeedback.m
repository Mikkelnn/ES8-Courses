% Demo: Variable Feedback.
% In this demo, the Variable Feedback Problem from the Leiden Workshop 
% 2005 is analyzed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
%
% Problem 5: Variable Feedback
% 
% ---------------------------------------------------------------------------
% Author  : Arne Hamann, Rafik Henia
% Contact : {arne,rafik}@ida.ing.tu-bs.de
% Date    : 2005-11-24
% Category: small
% ---------------------------------------------------------------------------
% 
% A) Description: 
% 
% The intention of this example is to compare the analysis accuracy (buffer 
% size and latency) for systems with feedback. The additional difficulty of 
% this example is the consideration of preemptions on CPU1.
% 
% Assumptions: CPU1 and CPU2 are scheduled according to the static priority 
% preemptive policy.
% 
% B) Contact:
% 
% Arne Hamann: arne@ida.ing.tu-bs.de
% Rafik Henia: rafik@ida.ing.tu-bs.de
% 
% C) References:

% Author(s): E. Wandeler
% Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
% ETH Zurich, Switzerland.
% All rights reserved.
% $Id: VariableFeedback.m 333 2006-02-17 16:59:14Z wandeler $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Event stream
% =========================================================================
% Stream 0:
p_S0 = 100;
j_S0 = 200;
d_S0 = 0;

% Stream 1:
p_S1 = 4;

% Processor speeds
% =========================================================================
% CPU0
b_CPU0 = 1;

% CPU1
b_CPU1 = 1;

% Task execution demands
% =========================================================================
ed_T0 = 20;
ed_T1 = 20;
ed_T2 = 40;
ed_T3 = 3;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Construct the input arrival curves
% =========================================================================
a0_S0 = rtcpjd(p_S0,j_S0,d_S0);
a0_S1 = rtcpjd(p_S1);

% Construct the service curves
% =========================================================================
b0_CPU0 = rtcfs(b_CPU0);
b0_CPU1 = rtcfs(b_CPU1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the arrival and service curves through a Greedy Processing 
% Components for Stream 1.
% =========================================================================
[a1_S1 b1_CPU1] = rtcgpc(a0_S1, b0_CPU1, ed_T3);

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components for Stream 0. 
% =========================================================================
[a1_S0 b1_CPU0] = rtcgpc(a0_S0, b0_CPU0, ed_T0);
[a2_S0 b2_CPU1] = rtcgpc(a1_S0, b1_CPU1, ed_T1);
[a3_S0 b2_CPU0] = rtcgpc(a2_S0, b1_CPU0, ed_T2);

% Compute the total delay through service curve convolution and by addition.
% =========================================================================
d_conv = rtcdel(a0_S0,b0_CPU0,ed_T0,b1_CPU1,ed_T1,b1_CPU0,ed_T2);
d_add = rtcdel(a0_S0,b0_CPU0,ed_T0) ...
      + rtcdel(a1_S0,b1_CPU1,ed_T1) ...
      + rtcdel(a2_S0,b1_CPU0,ed_T2);
d = min(d_conv,d_add);

% Compute the backlog buffer size at T2.
% =========================================================================
b = rtcbuf(a2_S0,b1_CPU0,ed_T2);

tTot = toc;

% Display the results.
% =========================================================================
disp(['End-To-End Delay of Stream 0 : ' num2str(d)])
disp(['Backlog Buffer at T2         : ' num2str(b)])
disp(['Analysis Time                : ' num2str(tTot) 's'])
