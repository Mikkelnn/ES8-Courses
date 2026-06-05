% Demo: In Car Radio Navigation System.
% In this demo, the system level end-to-end delays of architecture A of the 
% In Car Radio Navigation System Problem from the Leiden Workshop 2005 are 
% computed:
%
% Workshop on Distributed Embedded Systems
% Lorentz Center Leiden, November 21-24, 2005
% http://www.tik.ee.ethz.ch/~leiden05/
%
% Problem 2: In Car Radio Navigation System
% -------------------------------------------------------------------------
% Author  : Marcel Verhoef, Ernesto Wandeler, Paul Lieverse
% Contact : Marcel -dot- Verhoef -at- chess -dot- nl
% Date    : 2005-11-28
% Category: small 
% -------------------------------------------------------------------------
% 
% A) Description: 
% 
% The case study was inspired by a system architecture definition study for
% a distributed in-car radio navigation system. Such a system typically
% executes a number of independant but nevertheless concurrent applications
% that share a common platform. Each application has its own individual
% timeliness requirements that need to be satisfied. In this case study,
% three applications are specified using annotated UML sequence diagrams;
% ChangeVolume (A), AddressLookup (B) and HandleTMC (C). You are asked to
% evaluate the combinations A-C and B-C on five alternative distributed
% architectures. Informal definitions are provided for the deployment of
% these applications on those potential architectures. Hence, the case
% study should be seen as an exercise to perform simple design space
% exploration of a (soft) real-time system. The questions to answer are:
% - How do the proposed system architecture
%   compare in respect to system level end-to-end delays?
%   Are the WCRT requirements satisfied or not?
% - How robust is architecture A?
%   Where is the bottleneck of this architecture?
% - Architecture D is chosen for further investigation.
%   How should the processors be dimensioned while still
%   meeting the WCRT requirements
% 
% B) Contact:
% 
% Marcel -dot- Verhoef -at- chess -dot- nl
% 
% C) References:
% 
% An informal description (and MPA analysis results) can be found at
% http://www.mpa.ethz.ch The case study was first published at
% the ISOLA 2004 conference, the accepted paper can be found at
% http://www.esi.nl/site/projects/boderc/publications/isola_230904.pdf
% or http://www.cs.ru.nl/research/reports/info/ICIS-R05005.html
% Please note that the numbers on the MPA web-site differ from the ones
% in the original ISOLA paper! A more elaborate version of this
% paper is accepted for publication in the STTT journal (Springer,
% forthcoming in Q1-2006). The Timed Automata model can be found at
% http://www.cs.ru.nl/ita/publications/papers/martijnh/

% Author(s): E. Wandeler
% Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
% ETH Zurich, Switzerland.
% All rights reserved.
% $Id: InCarRadioNavigationSystem.m 333 2006-02-17 16:59:14Z wandeler $

% Note: in the following analysis, the end-to-end delays are computed as
% addition of the individual delays. Normally, one would compute the end-to
% -end delays through service curve convolutions. For the analyzed system
% configuration, the analysis results are equal for both methods. Since
% analysis through addition is considerably faster, this method was chosen.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Setup of system parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Task execution demands
% =========================================================================
MMIKeypress     = 100;  % x1000 instructions
MMIScreen       = 500;  % x1000 instructions
MMIForward      = 1;    % x1000 instructions
NAVLookup       = 5000; % x1000 instructions
NAVDecode       = 5000; % x1000 instructions
RADVolume       = 100;  % x1000 instructions
RADHandleTMC    = 1000; % x1000 instructions

% Message sizes
% =========================================================================
MSGTMCEncoded   = 512;  % bits
MSGTMCDecoded   = 512;  % bits
MSGKeypress     = 32;   % bits
MSGSetVolume    = 32;   % bits
MSGGetVolume    = 32;   % bits
MSGNAVResult    = 512;  % bits

% Event stream input rates
% =========================================================================
% All event streams are specified as strictly periodic event streams.
% The rate of the Change Audio Volume application is specified as 32Hz,
% which corresponds to a period of 31.25ms. Since the RTC Toolbox requires
% natural numbers for periods, we use a time quantum of 10ns. 
pVOL    = 3125;    % x10 ns
pADDR   = 100000;  % x10 ns
pTMC    = 300000;  % x10 ns

% Processor and bus speeds
% =========================================================================
cMMI    = 0.22;    % x 1000 instructions / 10 ns
cNAV    = 1.13;    % x 1000 instructions / 10 ns
cRADIO  = 0.11;    % x 1000 instructions / 10 ns
cBUS    = 0.72;    % x 1000 instructions / 10 ns


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Construct input curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Construct the input arrival curves
% =========================================================================
% All event streams are strictly periodic.
aVOL   = rtcpjd(pVOL,0,0);
aADDR  = rtcpjd(pADDR,0,0);
aTMC   = rtcpjd(pTMC,0,0);

% Construct the input service curves
% =========================================================================
% All resource are initially fully available.
bMMIi    = rtcfs(cMMI);
bNAVi    = rtcfs(cNAV);
bRADIOi  = rtcfs(cRADIO);
bBUSi    = rtcfs(cBUS);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze the change volume application.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Set initial resource availabilities.
% =========================================================================
bMMI   = bMMIi;
bRADIO = bRADIOi;
bBUS   = bBUSi;

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components.
% =========================================================================
[VOLa1 VOLb1] = rtcgpc(aVOL,  bMMI,   MMIKeypress);
[VOLa2 VOLb2] = rtcgpc(VOLa1, bBUS,   MSGSetVolume);
[VOLa3 VOLb3] = rtcgpc(VOLa2, bRADIO, RADVolume);
[VOLa4 VOLb4] = rtcgpc(VOLa3, VOLb2,  MSGGetVolume);
[VOLa5 VOLb5] = rtcgpc(VOLa4, VOLb1,  MMIScreen);

% Compute the individual delays at the Greedy Processing Components.
% =========================================================================
dVOL_1 = rtcdel(aVOL,  bMMI,   MMIKeypress)  ./ 100; % ms
dVOL_2 = rtcdel(VOLa1, bBUS,   MSGSetVolume) ./ 100; % ms
dVOL_3 = rtcdel(VOLa2, bRADIO, RADVolume)    ./ 100; % ms
dVOL_4 = rtcdel(VOLa3, VOLb2,  MSGGetVolume) ./ 100; % ms
dVOL_5 = rtcdel(VOLa4, VOLb1,  MMIScreen)    ./ 100; % ms

% Compute the total delay.
% =========================================================================
dVOLK2A = dVOL_4 + dVOL_5;

dVOLK2V = dVOL_1 + dVOL_2 + dVOL_3 + dVOL_4 + dVOL_5;

tVOL = toc;

% Display the results.
% =========================================================================
disp(['Volume Delay (K2A)  : ' num2str(dVOLK2A) 'ms'])
disp(['Volume Delay (K2V)  : ' num2str(dVOLK2V) 'ms'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze the handle TMC application with simultaneous volume change.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Set initial resource availabilities.
% =========================================================================
bMMI   = VOLb5;
bNAV   = bNAVi;
bRADIO = VOLb3;
bBUS   = VOLb4;

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components.
% =========================================================================
[TMCa1 TMCb1] = rtcgpc(aTMC,  bRADIO, RADHandleTMC);
[TMCa2 TMCb2] = rtcgpc(TMCa1, bBUS,   MSGTMCEncoded);
[TMCa3 TMCb3] = rtcgpc(TMCa2, bNAV,   NAVDecode);
[TMCa4 TMCb4] = rtcgpc(TMCa3, TMCb2,  MSGTMCDecoded);
[TMCa5 TMCb5] = rtcgpc(TMCa4, bMMI,   MMIScreen);

% Compute the individual delays at the Greedy Processing Components.
% =========================================================================
dTMCVOL_1 = rtcdel(aTMC,  bRADIO, RADHandleTMC)  ./ 100; % ms  
dTMCVOL_2 = rtcdel(TMCa1, bBUS,   MSGTMCEncoded) ./ 100; % ms
dTMCVOL_3 = rtcdel(TMCa2, bNAV,   NAVDecode)     ./ 100; % ms
dTMCVOL_4 = rtcdel(TMCa3, TMCb2,  MSGTMCDecoded) ./ 100; % ms
dTMCVOL_5 = rtcdel(TMCa4, bMMI,   MMIScreen)     ./ 100; % ms
 
% Compute the total delay.
% =========================================================================
dTMCVOL = dTMCVOL_1 + dTMCVOL_2 + dTMCVOL_3 + dTMCVOL_4 + dTMCVOL_5;

tTMCVOL = toc;

% Display the results.
% =========================================================================
disp(['TMC/VOL Delay       : ' num2str(dTMCVOL) 'ms'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze the address lookup application.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Set initial resource availabilities.
% =========================================================================
bMMI = bMMIi;
bNAV = bNAVi;
bBUS = bBUSi;

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components.
% =========================================================================
[ADDRa1 ADDRb1] = rtcgpc(aADDR,  bMMI,    MMIKeypress);
[ADDRa2 ADDRb2] = rtcgpc(ADDRa1, bBUS,    MSGKeypress);
[ADDRa3 ADDRb3] = rtcgpc(ADDRa2, bNAV,    NAVLookup);
[ADDRa4 ADDRb4] = rtcgpc(ADDRa3, ADDRb2,  MSGNAVResult);
[ADDRa5 ADDRb5] = rtcgpc(ADDRa4, ADDRb1,  MMIScreen);

% Compute the individual delays at the Greedy Processing Components.
% =========================================================================
dADDR_1 = rtcdel(aADDR,  bMMI,   MMIKeypress)  ./ 100; % ms
dADDR_2 = rtcdel(ADDRa1, bBUS,   MSGKeypress)  ./ 100; % ms
dADDR_3 = rtcdel(ADDRa2, bNAV,   NAVLookup)    ./ 100; % ms
dADDR_4 = rtcdel(ADDRa3, ADDRb2, MSGNAVResult) ./ 100; % ms
dADDR_5 = rtcdel(ADDRa4, ADDRb1, MMIScreen)    ./ 100; % ms

% Compute the total delay.
% =========================================================================
dADDR = dADDR_1 + dADDR_2 + dADDR_3 + dADDR_4 + dADDR_5;

tADDR = toc;

% Display the results.
% =========================================================================
disp(['Address Delay       : ' num2str(dADDR) 'ms'])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analyze the handle TMC application with simultaneous address lookup.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

% Set initial resource availabilities.
% =========================================================================
bMMI = ADDRb5;
bNAV = ADDRb3;
bRADIO = bRADIOi;
bBUS = ADDRb4;

% Compute the arrival and service curves through a chain of Greedy 
% Processing Components.
% =========================================================================
[TMCa1 TMCb1] = rtcgpc(aTMC,  bRADIO, RADHandleTMC);
[TMCa2 TMCb2] = rtcgpc(TMCa1, bBUS,   MSGTMCEncoded);
[TMCa3 TMCb3] = rtcgpc(TMCa2, bNAV,   NAVDecode);
[TMCa4 TMCb4] = rtcgpc(TMCa3, TMCb2,  MSGTMCDecoded);
[TMCa5 TMCb5] = rtcgpc(TMCa4, bMMI,   MMIScreen);

% Compute the individual delays at the Greedy Processing Components.
% =========================================================================
dTMCADDR_1 = rtcdel(aTMC,  bRADIO, RADHandleTMC)  ./ 100; % ms
dTMCADDR_2 = rtcdel(TMCa1, bBUS,   MSGTMCEncoded) ./ 100; % ms
dTMCADDR_3 = rtcdel(TMCa2, bNAV,   NAVDecode)     ./ 100; % ms
dTMCADDR_4 = rtcdel(TMCa3, TMCb2,  MSGTMCDecoded) ./ 100; % ms
dTMCADDR_5 = rtcdel(TMCa4, bMMI,   MMIScreen)     ./ 100; % ms

% Compute the total delay.
% =========================================================================
dTMCADDR = dTMCADDR_1 + dTMCADDR_2 + dTMCADDR_3 + dTMCADDR_4 + dTMCADDR_5;

tTMCADDR = toc;

% Display the results.
% =========================================================================
disp(['TMC/ADDR Delay      : ' num2str(dTMCADDR) 'ms'])

tTot = tTMCVOL + tTMCADDR + tVOL + tADDR;
disp(['Total Analysis Time : ' num2str(tTot) 's'])
