%function [valP, valF] = computeDynamicSlot( u, i , p, l, EF, EP, C, delta, H, Ltot, Lpre, ml)
%valP   time required to reach next time slot, with pending access
%valF   time required to reach next time slot, without pending access
%k      slot index
%u      up to u access request from phase under analysis
%i      up to i access requests from interferers
%p      number of access requests actually performed (for this iteration)
%l      number of interferences actually happened (for this iteration)
%EF     dynamic programming table, for non pending case, 2D
%EP     dynamic programming table, for pending case,2D 
%C      the constant time required to execute an access request
%delta  the length of the dynamic segment
%H      distance from the end of slot k to the beginning of slot k+1
%Ltot   the amount of interfering processing elements
%Lpre   the amount of interfering processing elements, that have a pending
%       access request
%ml     the length of a minislot
%
%EF and EP are 3 dimensional matrixes in the global scope. since for the
%functions only the layer with the appropriate slot k is required, only
%submit this layer to the function. 
%
%As an example, EP is created using zeros(4,4,4). To give only layer k to
%the function, use computeStaticSlot(k, u, i , p, EF(:,:,k), EP(:,:,k));
%Then EF and EP are common 2D matrixes with u and i as rows and columns.


function [valP, valF] = computeDynamicSlot( u, i , p, l , EF, EP, C, delta, H, Ltot, Lpre, ml)
    valFF = EF(u+1,i+1) + max( 0, delta - p*C - min(l, p*Ltot)*C) +H;
    valPF = EP(u+1,i+1) + max( 0, delta - (p+1)*C - min(l, (p+1)*Ltot+Lpre)*C) +H;

    if p == 0
        valFP = inf;
        valPP = inf;
    else
        valFP = EF(u+1,i+1) + max( 0, delta + ml - p*C - min(l, p*Ltot)*C);
        valPP = EP(u+1,i+1) + max( 0, delta + ml - (p+1)*C - min(l, (p+1)*Ltot+Lpre)*C);
    end
    
    valP = min(valFP, valPP);
    valF = min(valFF, valPF);
    
end