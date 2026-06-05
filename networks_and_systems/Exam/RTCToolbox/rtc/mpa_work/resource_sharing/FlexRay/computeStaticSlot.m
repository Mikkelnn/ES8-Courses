%function [valP, valF] = computeStaticSlot( u, i , p, EF, EP, C, delta, H)
%valP   time required to reach next time slot, with pending access
%valF   time required to reach next time slot, without pending access
%u      up to u access request from phase under analysis
%i      up to i access requests from interferers
%p      number of access requests actually performed (for this iteration)
%EF     dynamic programming table, for non pending case, 2D
%EP     dynamic programming table, for pending case,2D 
%C      the constant time required to execute an access request
%delta  the length of slot k
%H      distance from the end of slot k to the beginning of slot k+1
%
%EF and EP are 3 dimensional matrixes in the global scope. since for the
%functions only the layer with the appropriate slot k is required, only
%submit this layer to the function. 
%
%As an example, EP is created using zeros(4,4,4). To give only layer k to
%the function, use computeStaticSlot(k, u, i , p, EF(:,:,k), EP(:,:,k));
%Then EF and EP are common 2D matrixes with u and i as rows and columns.


function [valP, valF] = computeStaticSlot(u, i , p, EF, EP, C, delta, H)
    valFF = EF(u+1,i+1) + max( 0, delta - p*C) +H;
    valPF = EP(u+1,i+1) + max( 0, delta - (p+1)*C) +H;

    if p == 0
        valFP = inf;
        valPP = inf;
    else
        valFP = EF(u+1,i+1) + max(0, delta-p*C);
        valPP = EP(u+1,i+1) + max(0, delta-(p+1)*C);
    end
    
    valP = min(valFP, valPP);
    valF = min(valFF, valPF);
    
end