function [exectime, data] = sendShit(seg,dataL)

priority = 0;
T=0.4;
data=1;

switch seg
    case 1
        exectime = 0.001;
        
        msg = ttTryFetch('QJamesQ');
        if isempty(msg)
            ttCreateJob('sendFromBuffer', ttCurrentTime+T);
        else
            ttSendMsg([1 2], msg, 250, priority);
            ttCreateJob('sendFromBuffer', ttCurrentTime+T);
        end
    
    case 2
        exectime = -1;

end