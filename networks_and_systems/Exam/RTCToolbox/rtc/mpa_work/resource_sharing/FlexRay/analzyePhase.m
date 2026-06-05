%wcct = analzyePhase(t, exec, v_mu, theta, alpha, C, Ltot, ml, Lpre,
%assignment)
%
%t      the current time
%exec   the amount of computations
%v_mu     the amount of resoure accesses
%theta  structure of theta:
%       theta.assignments = array with length of slots, value is pu nr. 
%       theta.starting_times = array with starting times
%       theta.dyn_length = length of the dynamic segment
%       theta.L = length of the static segment
%C      the constant time required to execute an access request
%Ltot   the amount of interfering processing elements
%ml     the length of a minislot
%Lpre   the amount of interfering processing elements, that have a pending
%assignment     which processing element is the current phase assigned to

function [wcct] = analzyePhase(t, exec, v_mu, theta, alpha_rtc, C, Ltot, ml, Lpre, assignment)

%speed up if there is no access request;
if (v_mu == 0)
    wcct =  t + exec;
    %wcet = exec;
    return;
end


%how to choose k ?
%how to interpret alpha
%alpha = alpha_rtc;

%adapt the curve to the time scale.
alpha = zeros(1, ceil(t+theta.L+theta.dyn_length));
for rtcc=1:t+theta.L+theta.dyn_length
    alpha(rtcc) = rtcvalue(alpha_rtc, rtcc*1000, 1);
end

k = 50;

Delta = theta.dyn_length;
L = theta.L;
indices = find(theta.assignments == assignment);
sigma = theta.starting_times(indices);
sigma = [sigma L];
%phi = length(theta.assignments)+1;


if( indices(end)+1 <= length(theta.starting_times) )
    delta = theta.starting_times(indices+1) - theta.starting_times(indices);
else
    delta = theta.starting_times(indices(1:end-1)+1) - theta.starting_times(indices(1:end-1));
    delta = [delta theta.L - theta.starting_times(indices(end))];
end
    
delta = [delta theta.dyn_length];
phi = length(indices);

H = zeros(1,size(sigma,2));
for hc=1:phi+1
    if hc~=phi+1
        H(hc) = sigma(hc+1) - sigma(hc) - delta(hc);
    else
        H(hc) = sigma(1);
    end
end
    

EP = ones( v_mu+1,ceil(max(alpha))+1, k+1) .* inf;
EF = ones( v_mu+1,ceil(max(alpha))+1,k+1) .* inf;

ts_r = t - floor( t / (L + Delta)) * (L + Delta);

second_round = 0;
k_r = find(sigma >= ts_r);
if(isempty(k_r))
    k_r = 1;
    second_round = 1;
end
k_r = k_r(1);

%if(sigma(k_r+1) == ts_r)
%    k_r = k_r + 1;
%end
%if (k_r >= phi + 1)
%    k_r = 1;
%    second_round = 1;
%end

if ~second_round
    k_r = k_r - 1;
end

reachable = false;

for up = 0:v_mu
    for ip = 0:ceil((theta.dyn_length/ml))
        [EF(up+1,ip+1,k_r+1), EP(up+1,ip+1,k_r+1)] = initialize(ts_r, k_r, up, ip, sigma, delta, phi, C, H, Ltot, ml, second_round);
        %if the (first) next slot cannot be reached, finish here
        if ((EF(up+1,ip+1,k_r+1) <= exec) || (EP(up+1,ip+1,k_r+1) <= exec))
            reachable = true;
        end
    end
end

k = k_r -1;

if ~reachable
    if (ts_r < sigma(end)) %static slot
        wcct = t + v_mu*C + exec;
    else %dynamic slot
        wcct =  t + exec + v_mu*C + min( alpha(k+1), (v_mu )*(Ltot)) * C;
    end
    return;
end



%k = k_r;

while reachable
    reachable = false;
    k = k + 1;
    for u = 0:v_mu
        for i = 0:max(alpha)
            for p = 0:(v_mu - u)
                for l=0:(max(alpha)-i)
                    kp = mod(k, phi+1);
                    %if kp <= 0
                    %    kp = k;
                    %end
                    %k
                    %u
                    %i
                    %p
                    %l
                    if (kp < phi)                        
                        
                        [valP, valF] = computeStaticSlot(u, i , p, EF(:,:,k+1), EP(:,:,k+1), C, delta(kp+1), H(kp+1));
                        
                    else
                        [valP, valF] = computeDynamicSlot( u, i , p, l , EF(:,:,k+1), EP(:,:,k+1), C, delta(end), H(end), Ltot, Lpre, ml);
                    end
                    
                    if ((valP*10^16 <= exec*10^16) || (valF*10^16 <= exec*10^16)) %compensate numerical problems
                        %k
                        %valP
                        %valF
                        if (valP < 0 || valF < 0)
                            alarm = 1;
                        end
                        reachable = true;
                    end
                    %additional +1, because matlab indices start at 1
                    EP(u+p+1, i+l+1, k+1) = min( EP(u+p+1, i+l+1, k+1), valP);
                    EF(u+p+1, i+l+1, k+1) = min(EF(u+p+1, i+l+1, k+1), valF);
                end
            end
        end
    end
end

f = 0;


for up = 0:v_mu
    for ip = 0:max(alpha)
        f = max(f, finalize(k+k_r, up, ip, EF(:,:,k+1), EP(:,:,k+1), exec, phi, v_mu, C, alpha, Ltot, Lpre));
    end
end

t1 = floor( t / (L + Delta)) * (L + Delta);

if k_r+1+k <= length(delta) 
    t2 = sum(delta(k_r+1:k_r+1+k));
elseif k <= length(delta)
    t2 = sum(delta(k_r+1:end));
    t2 = t2 + sum(delta(1: k-(length(delta)-k_r+1)));
else
    t2 = (L + Delta) * floor( k/length(delta));
    k_t = mod(k, length(delta));
    t2 = t2 + sum(delta(k_r+1:end));
    t2 = t2 + sum(delta(1: k_t-(length(delta)-k_r+1)));
    
end

kp = mod(k, phi+1);
tf = floor(k/(phi+1)) * (L + Delta) + sigma(kp+1) + f;
wcct = t1 + tf + t2;

%if(second_round)
%    wcct = wcct + delta(end);
%end


end
