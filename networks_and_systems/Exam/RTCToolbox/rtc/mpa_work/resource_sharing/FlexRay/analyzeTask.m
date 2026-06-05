%[schedulable, wcct] = analyzeTask(S, alpha, theta, Ltot, C, ml, Lpre)
%S      N x 5 matrix: exec, v_mu, assignment, start_time (0 if subsequent),
%       deadline
%alpha  interference, represented as arrival curve. 
%theta  structure of theta:
%       theta.assignments = array with length of slots, value is pu nr. 
%       theta.starting_times = array with starting times
%       theta.dyn_length = length of the dynamic segment
%       theta.L = length of the static segment
%Ltot   the amount of interfering processing elements
%C      the constant time required to execute an access request
%ml     the length of a minislot
%Lpre   the amount of interfering processing elements, that have a pending

function [schedulable, wcct] = analyzeTask(S, alpha, theta, Ltot, C, ml, Lpre)




schedulable = true;
%t = 37.0725;
t=0;
for sbc = 1:size(S,1)
    if S(sbc, 4) == 0
        t_start = t; 
    else
        t_start = S(sbc, 4);
    end
%    S(sbc,1) = 0.0524;
%    S(sbc,2) = 2;
    [wcct] = analzyePhase(t_start, S(sbc,1) , S(sbc, 2), theta, alpha, C, Ltot, ml, Lpre, S(sbc,3));
    if(wcct < t)
        alarm = 1;
    end
    t =  wcct;
    
    if t >= S(sbc, 5)
        schedulable = false;
    end
 
end
wcct = t;
end