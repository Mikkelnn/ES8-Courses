% RTC_INSTALL   Install the Real-Time Calculus Toolbox.
%    RTC_INSTALL installs the Real-Time Calculus Toolbox.
%
%    See also RTC_UNINSTALL, RTC_INIT.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: rtc_install.m 348 2006-05-25 07:52:11Z wandeler $

disp('   ==============================================================');
disp('   Real-Time Calculus (RTC) Toolbox v1.0beta');
disp('   http://www.mpa.ethz.ch/rtctoolbox');
disp(' ');
disp('   Copyright (c) 2004-2007');
disp('   Computer Engineering and Networks Laboratory (TIK)');
disp('   ETH Zurich, Switzerland.');
disp('   All rights reserved.');
disp(' ');
disp('   Authors: Ernesto Wandeler, Nikolay Stoimenov, Lothar Thiele');
disp('   For support, write to: mpa@tik.ee.ethz.ch');
disp('   ==============================================================');
disp(' ');

matlabrelease = version('-release');
disp(['   Matlab Version: ', matlabrelease]);
disp(' ');

clear all
clear java

rtc_path = which('rtc_install');
rtc_path = strrep(rtc_path,'rtc_install.m','');
delimiter = rtc_path(length(rtc_path));
rtc_path = rtc_path(1:length(rtc_path)-1);
addpath(rtc_path);
addpath([rtc_path delimiter 'mpa']);
addpath([rtc_path delimiter 'curves']);
addpath([rtc_path delimiter 'operations']);
addpath([rtc_path delimiter 'mpa_demos']);
addpath([rtc_path delimiter 'mpa_work']);
addpath(genpath([rtc_path delimiter 'mpa_work' delimiter 'resource_sharing']));
success = savepath;

if (success == 1)
    warning('   The current MATLABPATH could not be saved to the pathdef.m \n   which was read on startup.\n   Please try to manually save the current MATLABPATH \n   For help type >> help savepath','');
    disp(' ');
    disp('   The Real-Time Calculus (RTC) Toolbox could not be installed completely.');
else
    disp('   The Real-Time Calculus (RTC) Toolbox was successfully installed.');  
end

rtc_init;

disp(' ')
disp('   Help on mpa functions: help mpa. Help on mpa demos: help mpa_demos');
