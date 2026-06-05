%f = finalize(k, u, i, EF, EP, exec, phi, v_mu, C, alpha, Ltot, Lpre)
%k      slot index
%u      up to u access request from phase under analysis
%i      up to i access requests from interferers
%EF     dynamic programming table, for non pending case, 2D
%EP     dynamic programming table, for pending case,2D 
%exec   the maximum amount of execution for the phase under analysis
%phi    the number of static slots assigned to the phase
%mu     the maximum amount of access request for the phase
%C      the constant time required to execute an access request
%alpha  the interfernce as arrival curve
%Ltot   the amount of interfering processing elements
%Lpre   the amount of interfering processing elements, that have a pending

function f = finalize(k, u, i, EF, EP, exec, phi, v_mu, C, alpha, Ltot, Lpre)

fF = 0;
fP = 0;

if (mod(k, phi+1) < phi)
    if EF(u+1,i+1) <= exec
        fF = exec - EF(u+1,i+1) + (v_mu -u)*C;
    end
    if EP(u+1,i+1) <= exec
        fP = exec - EP(u+1,i+1) + (v_mu - u+1)*C;
    end
else
    if EF(u+1,i+1) <= exec
        fF = exec - EF(u+1,i+1) + (v_mu - u + min(alpha(k+1) - i, (v_mu - u) * Ltot))*C;
    end
    if EP(u+1,i+1) <= exec
        fP = exec - EP(u+1,i+1) + (v_mu - u+1  + min( alpha(k+1) -i, (v_mu - u) * Ltot+Lpre)) * C;
    end
end

f = max(fP, fF);

end
