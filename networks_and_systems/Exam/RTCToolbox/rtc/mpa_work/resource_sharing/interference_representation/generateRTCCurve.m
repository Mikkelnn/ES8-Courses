function RTCcurve_upper = generateRTCCurve(upperCurve, period_of_task, cache_misses_worst_case)
%[RTCupperCurve] = generateRTCCurve(upperCurve, period_of_task, cache_misses_worst_case)
%
%This function computes an RTC uppper curve, give a vector representing the
%number of events (value) per time-window (index), the period of a
%reoccuring pattern (either a task or multiple tasks in a repeated static
%schedule) and the worst-case number of events per period.
%
%upperCurve     ... in ...  a vector representing the number of events in a time
%                           window. The index of the vector represents the
%                           time-window, while the value represents the actual
%                           number of events. As RTC curves start at a
%                           time-windows of 0, this vector will be shifted to
%                           the left by one position in the final results.
%period_of_task ... in ...  the period of a task. In case of a
%                           multi-task environment, the period of the static schedule
%cache_misses_worst_case ...The maximal number of events to occur in one
%                           period
%
%RTCupperCurve ...          The resulting RTC curve
%
%NOTE: The RTC toolbox is required for this function to work. Please
%download at www.mpa.ethz.ch.



%the aperiodic part is stored in lowerCurve and upperCurve
%and goes on to 2p
ASupper = zeros(2*period_of_task-1, 3);

PSupper = zeros(period_of_task-1,3);


%0 .. p => the aperiodic part is equal the computed arrival curve
for as_counter = 1:period_of_task
    ASupper(as_counter, 1) = as_counter-1;
    ASupper(as_counter, 2) = upperCurve(as_counter);
   % ASlower(as_counter, 1) = as_counter-1;
   % ASlower(as_counter, 2) = lowerCurve(as_counter);
  
end

%p .. 2p => choose the minimum/maximum for upper/lower curve
%           1) the arrival curve at position i
%           2) the sum of cache misses plus the arrival curve value at
%               position i-period.
for as_counter = period_of_task+1:2*period_of_task
    ASupper(as_counter, 1) = as_counter-1;
    value = max(upperCurve(as_counter), (sum(cache_misses_worst_case) + upperCurve(as_counter-period_of_task)));
    ASupper(as_counter, 2) = value;
%    value_l = min(lowerCurve(as_counter), (sum(cache_misses_best_case) + lowerCurve(as_counter-period_of_task)));
%    if(ASlower(as_counter-1,2) > value_l)
%        value_l = ASlower(as_counter-1,2);
%    end

    %ASlower(as_counter, 1) = as_counter-1;
    %ASlower(as_counter, 2) = value_l;
end

%compute the slopes accrodingly.
for as_counter = 1:2*period_of_task
   if(as_counter+1 < 2*period_of_task)
    %ASlower(as_counter, 3) = ASlower(as_counter+1, 2) - ASlower(as_counter,2);
    ASupper(as_counter, 3) = ASupper(as_counter+1, 2) - ASupper(as_counter,2);
   else
    %ASlower(as_counter, 3) = sum(cache_misses_best_case) + lowerCurve(as_counter+1-period_of_task) - ASlower(as_counter,2);
    ASupper(as_counter, 3) = sum(cache_misses_worst_case) + upperCurve(as_counter+1-period_of_task) - ASupper(as_counter,2);
   end
end

%the periodic part is k*sum(cache_misses_for_one_period)+the corresponding
%part of the lowerCurve/upperCurve
for ps_counter = 1:period_of_task
    %the values have to be offset, as the RTCcurve produces peaks otherwise
    %offset_upper = upperCurve(1);
%    offset_lower = lowerCurve(1);
    PSupper(ps_counter, 1) = ps_counter-1;
    PSupper(ps_counter, 2) = upperCurve(ps_counter) ;%- offset_upper;
    PSupper(ps_counter, 3) = upperCurve(ps_counter+1) - upperCurve(ps_counter);
    %PSlower(ps_counter, 1) = ps_counter-1;
    %PSlower(ps_counter, 2) = lowerCurve(ps_counter) - offset_lower;
    %PSlower(ps_counter, 3) = lowerCurve(ps_counter+1) - lowerCurve(ps_counter);
end



%create the arrival curves.
RTCcurve_upper = rtccurve(ASupper, PSupper, 2*period_of_task+1, ASupper(end,2), period_of_task);

%rtcplot(RTCcurve_upper, 'b', 1000, 'linewidth', 2);

end