% RTC_UNINSTALL   Uninstall the Real-Time Calculus Toolbox.
%    RTC_UNINSTALL uninstalls the Real-Time Calculus Toolbox.
%
%    See also RTC_INSTALL, RTC_INIT.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: rtc_uninstall.m 348 2006-05-25 07:52:11Z wandeler $

disp('   ==============================================================');
disp('   Real-Time Calculus (RTC) Toolbox v1.0beta');
disp('   http://www.mpa.ethz.ch/rtctoolbox');
disp(' ');
disp('   Copyright (c) 2004-2007');
disp('   Computer Engineering and Networks Laboratory (TIK)');
disp('   ETH Zurich, Switzerland.');
disp('   All rights reserved.');
disp(' ');
disp('   Authors: Ernesto Wandeler, Nikolay Stoimneov, Lothar Thiele');
disp('   For support, write to: mpa@tik.ee.ethz.ch');
disp('   ==============================================================');
disp(' ');

clear all
clear java

rtc_path = which('rtc_install');
rtc_path = strrep(rtc_path,'rtc_install.m','');
delimiter = rtc_path(length(rtc_path));
rtc_path = rtc_path(1:length(rtc_path)-1);
warning off;
rmpath(rtc_path);
rmpath([rtc_path delimiter 'mpa']);
rmpath([rtc_path delimiter 'curves']);
rmpath([rtc_path delimiter 'operations']);
rmpath([rtc_path delimiter 'mpa_demos']);
rmpath([rtc_path delimiter 'mpa_work']);
rmpath(genpath([rtc_path delimiter 'mpa_work' delimiter 'resource_sharing']));
success = savepath;

javarmpath([rtc_path delimiter 'rtc.jar']);
warning on;

if (success == 1)
    warning('   The current MATLABPATH could not be saved to the pathdef.m \n   which was read on startup.\n   Please try to manually save the current MATLABPATH \n   For help type >> help savepath','');
    disp(' ');
    disp('   The Real-Time Calculus (RTC) Toolbox could not be uninstalled.');
else
    disp('   The Real-Time Calculus (RTC) Toolbox was successfully uninstalled.');
end
