function [exectime, data] = receiver_code(seg,data)
global Nt

    msg = ttGetMsg(1);
    if(length(msg)>0)
       Nt = Nt+1
       ttAnalogOut(1,Nt)
    else
       %ttAnalogOut(1,ttCurrentTime);
    end
    exectime = 0.001;


    
