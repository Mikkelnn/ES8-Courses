% RTC_INIT   Initialize the Java Library of the Real-Time Calculus Toolbox.
%    RTC_INIT initializes the Java Library of the Real-Time Calculus Toolbox.
%
%    If the Real-Time Calculus Toolbox is not automatically 
%    initialized at startup-time, RTC_INIT must be executed 
%    manually prior to using the Real-Time Calculus Toolbox.
%
%    If a new Real-Time Calculus Library (rtc.jar) is installed
%    at runtime, RTC_INIT must be called to load this new library.
%
%    See also RTC_INSTALL, RTC_UNINSTALL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: rtc_init.m 352 2006-10-02 07:21:05Z wandeler $

%    clear java

rtc_path = which('rtc_install');
rtc_path = strrep(rtc_path,'rtc_install.m','');
delimiter = rtc_path(length(rtc_path));
rtc_path = rtc_path(1:length(rtc_path)-1);
javaaddpath([rtc_path delimiter 'rtc.jar']);

import ch.ethz.rtc.kernel.*;

disp('   Java Library of Real-Time Calculus (RTC) Toolbox initialized...')