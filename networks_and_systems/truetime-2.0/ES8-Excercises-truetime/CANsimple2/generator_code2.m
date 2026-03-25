function [exectime, data] = disturbance_code(seg,dataL)

networkNbr = 1;  % select a random network (1-3)
msg = [];                        % empty message
priority = 0;                    % highest priority
lambda=30;
u=rand();
msg = [ttCurrentTime]; 
T=u;
data=1;

switch seg
  case 1
    exectime = 0.005;
    ttPost('QJamesQ', msg); %Change to send to buffer
    ttCreateJob('generator_task',ttCurrentTime+T); %Creates next msg instant
  case 2
     exectime = -1;
end


