% Demo: Cyclic Dependencies.
% In this demo, the Cyclic Dependencies Problem from the Leiden Workshop 
% 2005 is analyzed:
%
% Problem 10: Cyclic Dependencies
% -------------------------------------------------------------------------
% Author  : Ernesto Wandeler, wandeler@tik.ee.ethz.ch
% Date    : 2005-11-24
% Category: small
% -------------------------------------------------------------------------
% 
% A) Description:
% 
% An event stream with jitter/burst is processed by a sequence of 3 tasks 
% running on two independent resources. On CPU1, a fixed-priority scheduler 
% is used to schedule T1 and T3. 
% 
% In scenario 1, T1 has higher priority than T3. In this scenario, 
% correlation effects may occur. I.e. depending on the execution time of T2
% and of the stream properties, it may happen, that T1 is never running 
% when T3 has to run, and that therefore T3 is never preempted by T1.
% 
% In scenario 2, T3 has higher priority than T1. In this scenario, the same
% correlation effects as in scenario 1 may occur, but additionally, there 
% exists a cyclyc dependency in the system.
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
% $Id: CyclicDependencies.m 328 2006-02-17 08:29:31Z wandeler $


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Event stream
% =========================================================================
p = 10;
j = 50;
d = 1;

% Processor speeds
% =========================================================================
cpu1 = 1;
cpu2 = 1;

% Task execution demands
% =========================================================================
wced1 = 1;
wced2 = 4;
wced3 = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct the input arrival curves
% =========================================================================
aui1 = rtcpjdu(p,j,d);
ali1 = rtcpjdl(p,j,d);

% Construct the service curves
% =========================================================================
bui1 = rtcfsu(cpu1);
bli1 = rtcfsl(cpu1);
bui2 = rtcfsu(cpu2);
bli2 = rtcfsl(cpu2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis of scenario 1    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Compute the arrival and service curves for the stream
% =========================================================================
[auo1 alo1 buo1 blo1] = rtcgpc(aui1, ali1, bui1, bli1, wced1);
[auo2 alo2 buo2 blo2] = rtcgpc(auo1, alo1, bui2, bli2, wced2);
[auo3 alo3 buo3 blo3] = rtcgpc(auo2, alo2, buo1, blo1, wced3);

% Compute the delay using both methods: addition and convolution
% =========================================================================
delay_conv = rtcdel(aui1,bli1,wced1,bli2,wced2,blo1,wced3);
delay_add = rtcdel(aui1,bli1,wced1) + rtcdel(auo1,bli2,wced2) + rtcdel(auo2,blo1,wced3);
delay = min(delay_conv, delay_add);

tTot = toc;

% Display the results
% =========================================================================
disp(['Scenarion 1: Priority(T3) < Priority(T1)'])
disp(['=================================================================='])
disp(['Delay         : ' num2str(delay)])
disp(['Analysis time : ' num2str(tTot) 's'])
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis of scenario 2    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This analysis requires a fixed-point calculation
tic;

% Compute the arrival and service curves for the stream
% =========================================================================
% Set start values for fixed-point calculation
buo3 = bui1;
blo3 = bli1;
buo3last = rtcuplus(buo3); % unary plus makes a clone!
blo3last = rtcuplus(blo3);

% Iterate up to 20 times
for i = [1:20]    
    [auo1 alo1 buo1 blo1] = rtcgpc(aui1, ali1, buo3, blo3, wced1);
    [auo2 alo2 buo2 blo2] = rtcgpc(auo1, alo1, bui2, bli2, wced2);
    [auo3 alo3 buo3 blo3] = rtcgpc(auo2, alo2, bui1, bli1, wced3);
    if ((buo3 == buo3last) && (blo3 == blo3last))
        break
    end
    buo3last = rtcuplus(buo3); % unary plus makes a clone!
    blo3last = rtcuplus(blo3);
end 

% Compute the delay using both methods: addition and convolution
% =========================================================================
delay_conv = rtcdel(aui1,blo3,wced1,bli2,wced2,bli1,wced3);
delay_add = rtcdel(aui1,blo3,wced1) + rtcdel(auo1,bli2,wced2) + rtcdel(auo2,bli1,wced3);
delay = min(delay_conv, delay_add);

tTot = toc;

% Display the results
% =========================================================================
disp(['Scenarion 2: Priority(T3) > Priority(T1) (fixed-point calculation)'])
disp(['=================================================================='])
disp(['Delay         : ' num2str(delay)])
disp(['Analysis time : ' num2str(tTot) 's'])
if (((buo3 == buo3last) & (blo3 == blo3last)))
    disp(['Fixed point reached after ' num2str(i) ' iterations.'])
else 
    disp(['Fixed point NOT reached after ' num2str(i) ' iterations.'])
end    
disp(' ')