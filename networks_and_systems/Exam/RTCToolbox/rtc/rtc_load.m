%-------------------------------------------------------------------------
% Init the RTC toolbox.
% Replace "matlabroot" by "pwd" if you want to use this script in a
% directory where the toolbox is located.
%
% modification history:
% 2007-02-26: created
% 2007-07-05: added -end option to all addpath commands
%             added note about pwd
% 2007-08-15: modification for new matlab toolbox directory structure
%-------------------------------------------------------------------------

current_path = path;
if (isempty(strfind(path, 'rtc')))
  warning('off', 'MATLAB:dispatcher:nameConflict');
  addpath([pwd], '-end');
  addpath([pwd, '/curves'], '-end');
  addpath([pwd, '/mpa'], '-end');
  addpath([pwd, '/mpa_demos'], '-end');
  addpath([pwd, '/operations'], '-end');
  rtc_init;
  warning('on', 'MATLAB:dispatcher:nameConflict');
end
