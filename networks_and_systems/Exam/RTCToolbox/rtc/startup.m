% STARTUP   Initialize the Real-Time Calculus Toolbox at startup.
%    STARTUP initializes the Real-Time Calculus Toolbox at startup.
%
%    If the Real-Time Calculus Toolbox is not automatically 
%    initialized at startup-time, RTC_INIT must be executed 
%    manually prior to using the Real-Time Calculus Toolbox.
%
%    If a new Real-Time Calculus Library (rtc.jar) is installed
%    at runtime, RTC_INIT must be called to load this new library.
%
%    See also RTC_INIT, RTC_INSTALL, RTC_UNINSTALL.

%    Author(s): E. Wandeler
%    Copyright 2004-2006 Computer Engineering and Networks Laboratory (TIK) 
%    ETH Zurich, Switzerland.
%    All rights reserved.
%    $Id: startup.m 379 2007-03-20 20:38:44Z nikolays $

rtc_init;