% Demo: AND-OR Activation.
% In this demo, the AND-OR Activation Problem from the Leiden Workshop
% 2005 is analyzed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
% 
% Problem 3: And-Or Activation
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
% A single task running on a static priority preemptive scheduled resource 
% is activated by two event streams. We are interested in two scenarios:
% 
% 1) The task is AND-activated, i.e. an event must be available on both 
%    streams to activate the task.
% 2) The task is OR-activated, i.e. each event on both streams activates 
%    the task.
% 
% The intention of this example is two compare the different approaches 
% with respect to modeling effort and analysis accuracy (response time, 
% backlog, output behavior) for tasks with multiple input ports.
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
% $Id: AndOrActivation.m 347 2006-05-25 05:58:34Z wandeler $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters for OR-activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Event streams
% =========================================================================
% Stream 0:
p_S0 = 100;
j_S0 = 20;

% Stream 1:
p_S1 = 150;
j_S1 = 60;

% Processor speeds
% =========================================================================
% CPU0
b_CPU0 = 1;

% Task execution demands
% =========================================================================
% T0
ed_T0 = 40;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a_S0 = rtcpjd(p_S0,j_S0);
a_S1 = rtcpjd(p_S1,j_S1);
b_CPU0 = rtcfs(b_CPU0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Compute the OR-activation
% =========================================================================
a_OR = rtcor(a_S0,a_S1);

% Compute the output behavior of T0
% =========================================================================
[aT0 bT0] = rtcgpc(a_OR, b_CPU0, ed_T0);

% Compute the delay and backlog on T0
% =========================================================================
d_OR = rtcdel(a_OR,b_CPU0,ed_T0);
b_OR = rtcbuf(a_OR,b_CPU0,ed_T0);


tTot = toc;

% Display the results
% =========================================================================
plotaxis = [0 1000 0 20];
subplot(2,1,1)
rtcplot(a_OR,'r',aT0,'b:',plotaxis)
title('Input (red) and Output (blue dashed) behavior of T0 with OR-activation.')
disp('OR-Activation: ')
disp('======================================================================')
disp(['Delay at T0     : ' num2str(d_OR)])
disp(['Backlog at T0   : ' num2str(b_OR)])
disp(['Analysis time   : ' num2str(tTot) 's'])
disp(['Output behavior :'])
disp('The output behavior of T0 with OR-activation is depicted in the upper')
disp('graph of the figure. As one can see, the output behavior could only be')
disp('approximated with a PJD-Curve, as it is more complex.')
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters for AND-activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
% Event streams
% =========================================================================
% Stream 2:
p_S2 = 100;
j_S2 = 10;
d_S2 = 0;

% Stream 3:
p_S3 = 100;
j_S3 = 190;
d_S3 = 20;

% Processor speeds
% =========================================================================
% CPU1
b_CPU1 = 1;

% Task execution demands
% =========================================================================
% T1
ed_T1 = 40;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

a_S2 = rtcpjd(p_S2,j_S2,d_S2);
a_S3 = rtcpjd(p_S3,j_S3,d_S3);
b_CPU1 = rtcfs(b_CPU1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Compute the AND-activation
% =========================================================================
a_AND = rtcand(a_S2,a_S3);

% Compute the output behavior of T1
% =========================================================================
[aT1 bT1] = rtcgpc(a_AND, b_CPU1, ed_T1);

% Compute the delay and backlog on T1
% =========================================================================
% d_AND = rtcdel(a_AND,b_CPU1,ed_T1);
% d_AND_S2 = d_AND + rtch(a_S2,a_S3);
% d_AND_S3 = d_AND + rtch(a_S3,a_S2);

bt_S2 = rtcminconv(rtctimes(a_S3(2), ed_T1), b_CPU1(2));
bt_S3 = rtcminconv(rtctimes(a_S2(2), ed_T1), b_CPU1(2));
d_AND_S2 = rtcdel(a_S2(1), bt_S2, ed_T1);
d_AND_S3 = rtcdel(a_S3(1), bt_S3, ed_T1);
b_AND = rtcbuf(a_AND,b_CPU1,ed_T1) * 2 + max(rtcv(a_S3,a_S2), rtcv(a_S2,a_S3));

tTot = toc;

% Display the results
% =========================================================================
plotaxis = [0 500 0 10];
subplot(2,1,2)
rtcplot(a_AND,'r',aT1,'b:',plotaxis)
title('Input (red) and Output (blue dashed) behavior of T1 with AND-activation.')
disp('AND-Activation: ')
disp('======================================================================')
disp(['Delay of S2 at T1 : ' num2str(d_AND_S2)])
disp(['Delay of S3 at T1 : ' num2str(d_AND_S3)])
disp(['Backlog at T1     : ' num2str(b_AND)])
disp(['Analysis time     : ' num2str(tTot) 's'])
disp(['Output behavior   :'])
disp('The output behavior of T1 with AND-activation is depicted in the lower')
disp('graph of the figure.')