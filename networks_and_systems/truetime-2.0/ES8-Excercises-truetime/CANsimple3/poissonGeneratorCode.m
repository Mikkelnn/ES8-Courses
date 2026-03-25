function [exectime, data] = poissonSend(seg,dataL)

networkNbr = 1;  % select a random network (1-3)
msg = [];                        % empty message
priority = 0;                    % highest priority
lambda=30;
u=randi([0,100])*0.0002;
msg = [ttCurrentTime]; 
T=u;
data=1;

switch seg
  case 1
    exectime = 0.0001;
    ttSendMsg([1 3], msg, 1000, priority);
    ttCreateJob('poissonGenerator',ttCurrentTime+T);
  case 2
     exectime = -1;
end