% Demo: Greedy Shapers.
% In this demo, the Greedy Shapers Problem from the Leiden Workshop 2005
% is analyzed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
%
% Problem 17: Greedy Shapers
% ---------------------------------------------------------------------------
% Author  : Ernesto Wandeler, wandeler@tik.ee.ethz.ch
% Date    : 2005-11-28
% Category: small
% ---------------------------------------------------------------------------
% 
% A) Description:
% 
% Three streams are processed on two processors. Due to the burstiness of
% the stream S1, it may happen that task T2 does not get processed for a
% considerable amount of time on T2. When T2 eventually gets processed, many
% of the buffered events may be processed almost at once, leading to a large
% burst on the output event stream of T2. Without a shaper, this large burst
% would appear at CPU2, and consequently, it could then happen that T4 would
% not be processed for a considerable amount of time, leading to a large delay
% on stream S3. To prevent this, the output stream of T2 gets shaped with a
% greedy shaper.
% 
% Greedy shapers have some nice properties that make them interesting for
% application. Most of all, if the shaping curve of the greedy shapers equals
% the input curve of the event stream (as it is the case in this example), then the
% shaper does not add any delay! I.e. the end-to-end delay for processing S2 on
% T2 and shaping it on S is the same as for processing it on T2 only. Furthermore,
% if the shaper and the buffer in front of T2 access the same shared memory, then
% adding the shaper does not lead to any additional memory requirements!
% 
% Assumptions:
% 1) All task are preemptive and independent.
% 
% B) Contact:
% 
% Ernesto Wandeler, wandeler@tik.ee.ethz.ch
% Lothar Thiele, thiele@tik.ee.ethz.ch
% 
% C) References:
% The very similar example is analyzed in the following paper:
% E. Wandeler, A. Maxiaguine, L. Thiele.
% "Performance Analysis of Greedy Shapers in Real-Time Systems".
% In Design, Automation and Test in Europe (DATE),  Munich, Germany, March, 2006.
% Accepted for Publication

% Author(s): E. Wandeler
% Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
% ETH Zurich, Switzerland.
% All rights reserved.
% $Id: GreedyShapers.m 336 2006-02-21 12:46:53Z wandeler $

% Result plot limit
% =========================================================================
limit = [0 200 0 20];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Event streams
% =========================================================================
% Stream S1:
p_S1 = 100;
j_S1 = 1000;
d_S1 = 1;

% Stream S2:
p_S2 = 10;
j_S2 = 10;
d_S2 = 1;

% Stream S3:
p_S3 = 10;
j_S3 = 0;
d_S3 = 0;

% Greedy shaper output
% =========================================================================
p_GS = 10;
j_GS = 10;
d_GS = 1;


% Processor speeds
% =========================================================================
cpu1 = 1;
cpu2 = 1;

% Task execution demands
% =========================================================================
ed_T1 = 5;
ed_T2 = 1;
ed_T3 = 5;
ed_T4 = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Construct the input arrival curves
% =========================================================================
a_S1 = rtcpjd(p_S1,j_S1,d_S1);
a_S2 = rtcpjd(p_S2,j_S2,d_S2);
a_S3 = rtcpjd(p_S3,j_S3,d_S3);

% Construct the shaper curve
% =========================================================================
%sigma  = rtcpjdu(p_GS,j_GS,d_GS);
sigma  = rtccurve([[0.0 0.0 1.0];[1.1111111111111112 1.1111111111111112 0.1]]);

% Construct the service curves
% =========================================================================
b_CPU1 = rtcfs(cpu1);
b_CPU2 = rtcfs(cpu2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis without Greedy Shaper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the arrival and service curves for Stream S1
% =========================================================================
[a1_S1 b1_CPU1] = rtcgpc(a_S1, b_CPU1, ed_T1);

% Compute the arrival and service curves for Stream S2
% =========================================================================
[a1_S2 b2_CPU1] = rtcgpc(a_S2, b1_CPU1, ed_T2);
[a3_S2 b1_CPU2] = rtcgpc(a1_S2, b_CPU2, ed_T3);

% Compute the arrival and service curves for Stream S3
% =========================================================================
[a1_S3 b2_CPU2] = rtcgpc(a_S3, b1_CPU2, ed_T4);

% Compute the delays
% =========================================================================
d_S1_n = rtcdel(a_S1, b_CPU1, ed_T1);
d_S2_n = rtcdel(a_S2, b1_CPU1, ed_T2, b_CPU2, ed_T3);
d_S3_n = rtcdel(a_S3, b1_CPU2, ed_T4);

% Compute the buffers
% =========================================================================
b_CPU1_n = rtcbuf(rtcplus(a_S1, a_S2), b_CPU1);
b_CPU2_n = rtcbuf(rtcplus(a1_S2, a_S3), b_CPU2);

% Display the results
% =========================================================================
disp(['Without Greedy Shaper:'])
disp(['======================'])
disp(['Delay Stream 1 : ' num2str(d_S1_n)])
disp(['Delay Stream 2 : ' num2str(d_S2_n)])
disp(['Delay Stream 3 : ' num2str(d_S3_n)])
disp(['Buffer CPU 1   : ' num2str(b_CPU1_n)])
disp(['Buffer CPU 2   : ' num2str(b_CPU2_n)])
disp(' ')

subplot(3,4,1);
hold off;
rtcplot(a_S1,'r',rtcrdivide(b_CPU1, ed_T1),'b',limit);
title('Stream 1: Input to T1')
subplot(3,4,4);
hold off;
rtcplot(a1_S1,'r',limit);
title('Stream 1: Output')
subplot(3,4,5);
hold off;
rtcplot(a_S2,'r',rtcrdivide(b1_CPU1, ed_T2),'b',limit);
title('Stream 2: Input to T2')
subplot(3,4,7);
hold off;
rtcplot(a1_S2,'r',rtcrdivide(b_CPU2, ed_T3),'b',limit);
title('Stream 2: Input to T3')
subplot(3,4,8);
hold off;
rtcplot(a3_S2,'r',limit);
title('Stream 2: Output')
subplot(3,4,11);
hold off;
rtcplot(a_S3,'r',rtcrdivide(b1_CPU2, ed_T4),'b',limit);
title('Stream 3: Input to T4')
subplot(3,4,12);
hold off;
rtcplot(a1_S3,'r',limit);
title('Stream 3: Output')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis with Greedy Shaper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute the arrival and service curves for Stream S1
% =========================================================================
[a1_S1 b1_CPU1] = rtcgpc(a_S1, b_CPU1, ed_T1);

% Compute the arrival and service curves for Stream S2
% =========================================================================
[a1_S2 b2_CPU1]  = rtcgpc(a_S2, b1_CPU1, ed_T2);
a2_S2           = rtcgsc(a1_S2, sigma);
[a3_S2 b1_CPU2] = rtcgpc(a2_S2, b_CPU2, ed_T3);

% Compute the arrival and service curves for Stream S1
% =========================================================================
[a1_S3 b2_CPU2] = rtcgpc(a_S3, b1_CPU2, ed_T4);

% Compute the delays
% =========================================================================
d_S1_s = rtcdel(a_S1, b_CPU1, ed_T1);
d_S2_s = rtcdel(a_S2, b1_CPU1, ed_T2, sigma, 1, b_CPU2, ed_T3);
d_S3_s = rtcdel(a_S3, b1_CPU2, ed_T4);

% Compute the buffers
% =========================================================================
b_CPU1_s = rtcbuf(rtcplus(a_S1, a_S2), b_CPU1);
b_CPU2_s = rtcbuf(rtcplus(a2_S2, a_S3), b_CPU2);


% Display the results
% =========================================================================
disp(['With Greedy Shaper:'])
disp(['======================'])
disp(['Delay Stream 1 : ' num2str(d_S1_s)])
disp(['Delay Stream 2 : ' num2str(d_S2_s)])
disp(['Delay Stream 3 : ' num2str(d_S3_s)])
disp(['Buffer CPU 1   : ' num2str(b_CPU1_s)])
disp(['Buffer CPU 2   : ' num2str(b_CPU2_s)])
disp(' ')

subplot(3,4,1);
hold on;
rtcplot(a_S1,'r',rtcrdivide(b_CPU1, ed_T1),'b',limit,'linewidth',2);
title('Stream 1: Input to T1')
subplot(3,4,4);
hold on;
rtcplot(a1_S1,'r',limit,'linewidth',2);
title('Stream 1: Output')
subplot(3,4,5);
hold on;
rtcplot(a_S2,'r',rtcrdivide(b1_CPU1, ed_T2),'b',limit,'linewidth',2);
title('Stream 2: Input to T2')
subplot(3,4,6);
rtcplot(a1_S2,'r',sigma,'g',limit,'linewidth',2);
title('Stream 2: Input to Shaper')
subplot(3,4,7);
hold on;
rtcplot(a2_S2,'r',rtcrdivide(b_CPU2, ed_T3),'b',limit,'linewidth',2);
title('Stream 2: Input to T3')
subplot(3,4,8);
hold on;
rtcplot(a3_S2,'r',limit,'linewidth',2);
title('Stream 2: Output')
subplot(3,4,11);
hold on;
rtcplot(a_S3,'r',rtcrdivide(b1_CPU2, ed_T4),'b',limit,'linewidth',2);
title('Stream 3: Input to T4')
subplot(3,4,12);
hold on;
rtcplot(a1_S3,'r',limit,'linewidth',2);
title('Stream 3: Output')

disp('Information to the figure: ')
disp(' - arrival curves are red')
disp(' - service curves are blue and scaled down to event units')
disp(' - the shaper curve is green')
disp(' - curves for the greedy shaper scenario are bold')
disp(' - curves for the scenario without greedy shaper are fine')


