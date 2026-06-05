%[valF, valP] = initialize(ts_r, k_r, u, i, sigma, delta, phi, C, H, Ltot,
%ml)
%valP   time required to reach next time slot, with pending access
%valF   time required to reach next time slot, without pending access
%k      slot index
%u      up to u access request from phase under analysis
%i      up to i access requests from interferers 
%sigma  starting times of slots assigned to the phase
%delta  the length of slot k
%phi    the number of static slots assigned to the phase
%C      the constant time required to execute an access request
%H      distance from the end of slot k to the beginning of slot k+1
%Ltot   the amount of interfering processing elements
%ml     the length of a minislot


function [valF, valP] = initialize(ts_r, k_r, u, i, sigma, delta, phi, C, H, Ltot, ml, second_round)
    
    if (ts_r <= sigma(1) || (k_r > 0 && ts_r >= (sigma(k_r+1 -1) + delta(k_r+1 -1)))) && ts_r < sigma(end)
        valF = sigma(k_r+1) - ts_r;
        if u == 0
            valP = inf;
        else
            valP = 0;
        end
        
    elseif(k_r > 0  && ts_r < sigma(end) )
        valF = max(0, sigma(k_r+1 -1) + delta(k_r+1 -1) -ts_r - u*C) + H(k_r+1 -1);
        if u == 0
            valP = inf;
        else
            valP = max(0, sigma(k_r+1 -1) + delta(k_r+1 -1) -ts_r - u*C);
        end
    else
       valF = max(0, (sigma(end)+delta(end)) - ts_r - u*C - min(i, u*Ltot)*C + H(end));
       %valF = max(0, sigma(k_r+1 -1) + delta(k_r+1 -1) -ts_r - u*C - min(i, u*Ltot)*C) + H(k_r+1 -1); 
       if u == 0
            valP = inf;
       else
            valP =max(0,(sigma(end)+delta(end)) - ts_r + ml - u*C - min(i, u*Ltot)*C); 
       end
    end


end