function [exectime, data] = receiver_code(seg,data)
global delays
    msg = ttGetMsg(1);
    if(length(msg)>0)
       currentDelay = ttCurrentTime - msg;
       ttAnalogOut(1,currentDelay);
       if isempty(delays)
           delays(1)=currentDelay;
       else
           delays = [delays currentDelay];
       end
    else
       %ttAnalogOut(1,ttCurrentTime);
    end
    exectime = 0.001;


    
