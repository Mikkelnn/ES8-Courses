% Demo: Intra Context Information.
% In this demo, the Intra Context Information Problem from the Leiden 
% Workshop 2005 is analyzed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
%
% Problem 4: Intra-Context Information
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
% In the example system the task T0 decodes a MPEG stream. Its core execution time, i.e. assuming no
% interrupts, varies according to the decoded frame type. Intraframes (I) take the longest to decode.
% Predictive frames (P) are decoded faster than intraframes but slower than bi-directional frames (B).
% 
% In the given example a cyclic pattern of decoded frames is assumed. The difficulty of this example
% is to exploit this information for worst-case analysis.
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
% $Id: IntraContextInformation.m 333 2006-02-17 16:59:14Z wandeler $

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Event streams
% =========================================================================
% Stream S0:
% Pattern: BBPBBI => worst-case pattern is IBBPBB, 
p_S0 = 100;

% Stream S1:
p_S1 = 500;

% Processor speeds
% =========================================================================
cpu0 = 1;

% Task execution demands
% =========================================================================
ed_T0_I = 80;
ed_T0_P = 40;
ed_T0_B = 20;
ed_T1   = 200;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Construct the input arrival curves
% =========================================================================
% This curve is a resource-based arrival curve, that can be obtained by 
% transformation of an event-based arrival curve with a workload curve.
aui_S0 = rtccurve([[0 ed_T0_I 0];...
                [p_S0 ed_T0_I+ed_T0_B 0];...
                [2*p_S0 ed_T0_I+ed_T0_B+ed_T0_B 0];...
                [3*p_S0 ed_T0_I+ed_T0_B+ed_T0_B+ed_T0_P 0];...
                [4*p_S0 ed_T0_I+ed_T0_B+ed_T0_B+ed_T0_P+ed_T0_B 0];...
                [5*p_S0 ed_T0_I+ed_T0_B+ed_T0_B+ed_T0_P+ed_T0_B+ed_T0_B 0]], ...
                0, 6*p_S0, ed_T0_I+ed_T0_B+ed_T0_B+ed_T0_P+ed_T0_B+ed_T0_B);

aui_S1 = rtcpjdu(p_S1);

% Construct the service curve
% =========================================================================
bli_CPU0 = rtcfsl(cpu0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To answer the design question, we do not need to analyze full GPCs.

% Compute the lower servie curve after processing T0
% =========================================================================
blo1_CPU0 = rtcmaxconv(rtcminus(bli_CPU0, aui_S0), 0);

% Compute the delay of T1
% =========================================================================
d_T1 = rtcdel(aui_S1,blo1_CPU0,ed_T1);

tTot = toc;

% Display the results
% =========================================================================
disp(['Delay of T1   : ' num2str(d_T1)])
disp(['Analysis time : ' num2str(tTot) 's'])
disp(' ')

plotaxis = [0 1000 0 500];
figure(1)
subplot(2,1,1)
rtcplot(aui_S0,'r',bli_CPU0,'b',plotaxis)
title('Upper arrival (red) and lower service (blue) input curves to T0')
subplot(2,1,2)
rtcplot(blo1_CPU0,'b',rtctimes(aui_S1, ed_T1),'r',plotaxis)
title('Upper arrival (red) and lower service (blue) input curves to T1')
rtcploth(rtctimes(aui_S1, ed_T1), blo1_CPU0)