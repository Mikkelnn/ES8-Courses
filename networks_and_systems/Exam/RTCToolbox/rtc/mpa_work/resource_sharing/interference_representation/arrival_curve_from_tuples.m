
function [RTCcurve_lower, RTCcurve_upper] = arrival_curve_from_tuples(config_file)


%base_dir = 'C:\Documents and Settings\andrschr\My Documents\MySvn\paper\schedulability\implementation';
%base_dir = '/Users/andreas/Documents/work/ETH/mysvn/paper/schedulability/implementation';

%read the input data
%base_dir= '/home/andreas/svn_local/andrschr/paper/schedulability/implementation';

%config_file = sprintf('%s/config.data', base_dir);


[A D H] = importdata(config_file, ' ',13);
number_of_super_blocks = A.data(1);
period_of_task = A.data(2);
cache_misses_best_case = A.data(3:3+number_of_super_blocks-1);
cache_misses_worst_case = A.data(3+number_of_super_blocks:3+2*number_of_super_blocks-1);
communication_time_per_miss = 1;% A.data(3+2*number_of_super_blocks);
execution_time_worst = A.data(3+2*number_of_super_blocks+1:3+3*number_of_super_blocks);
execution_time_best = A.data(3+3*number_of_super_blocks+1:3+4*number_of_super_blocks);

upperCurve = zeros(1, 2*period_of_task);
lowerCurve = Inf*ones(1, 2*period_of_task);

%loop over all possible distances. The distance is the
%number of subsequent superblocks considered.
%distance = 0 means means, that each superblocks values
%are considered for the tuples,
%distance = num_blocks means, that the sum over all superblocks
%is considered for constructing the tuples
for distance = 0:2*number_of_super_blocks-1 
    tuples = computeTuples(distance, execution_time_best, execution_time_worst, communication_time_per_miss, cache_misses_best_case, cache_misses_worst_case, number_of_super_blocks, period_of_task);
    %for each set of tuples, update the curves
    [upperCurve lowerCurve] = computeCurves(tuples, upperCurve, lowerCurve, distance);

end


%smooth the arrival curves:
    
%for upper curve, go on from first item and replace any succeeding
%value that is lower than its preceeding by the preceeding.
last = 0;
for curve_counter = 1:length(upperCurve)
    if last > upperCurve(curve_counter)
        upperCurve(curve_counter) = last;
    elseif last <= upperCurve(curve_counter)
        last = upperCurve(curve_counter);
    end
end


%for lower curve, check the minimum value and its index.
%up to that index, replace all values with the minimum.
%the find the next and replace all values from the last
%considered index to the newly computed minimums index. 
lower_indexes = (find(lowerCurve ~= Inf));
[lower_value lower_index] = max(lowerCurve(lower_indexes));
lowerCurveTemp = lowerCurve;
last_index = 1;
%%construct the lower Curve
while(true)
    [current_value current_index] = min(lowerCurveTemp);
    lowerCurve(last_index+current_index-1:end) = current_value;
    lowerCurveTemp(1:current_index) = [];
    last_index = last_index + current_index;
    if(current_value == lower_value)
        lowerCurve(last_index-current_index:end) = current_value;
        break;
    end

end


%STARTING HERE:
%compute an infinte curve from values for \Delta between 0 and 2p

%the curve has different shapes at different sections:
%from 0 .. period_of_task: shape as computed above
%from p .. 2*period_of_task: the maximum between the shape computed and
%           the sum(cache_misses) + alpha(delta-period_of_task)
%           (minimum for the lower curve)
%from 2*p...k*p: k*sum(cache_misses) + alpha(delta-period_of_task)


%%TESTDATA, to check the RTCcurve
k = 10;

overall_upperCurve = zeros(1, k*period_of_task);
overall_lowerCurve = zeros(1, k*period_of_task);

for i=1:k*(period_of_task)
    if(i <= period_of_task)
        overall_upperCurve(i) = (upperCurve(i));
        overall_lowerCurve(i) = lowerCurve(i);
    elseif (i > period_of_task && i <= 2*period_of_task)
        overall_upperCurve(i) = max(upperCurve(i), upperCurve(i-period_of_task)+sum(cache_misses_worst_case));
        overall_lowerCurve(i) = min(lowerCurve(i), lowerCurve(i-period_of_task)+sum(cache_misses_best_case));
    else
        iteration = floor(i/period_of_task);
        if(i-iteration*period_of_task == 0)
           overall_upperCurve(i) = upperCurve(period_of_task)+(iteration-1)*sum(cache_misses_worst_case);
           overall_lowerCurve(i) = lowerCurve(period_of_task)+(iteration-1)*sum(cache_misses_best_case);
       else
            overall_upperCurve(i) = upperCurve(i-iteration*period_of_task)+iteration*sum(cache_misses_worst_case);
            overall_lowerCurve(i) = lowerCurve(i-iteration*period_of_task)+iteration*sum(cache_misses_best_case);   
        end

    end

end


%generate an MPA compatible representation:

%the aperiodic part is stored in lowerCurve and upperCurve
%and goes on to 2p
ASupper = zeros(2*period_of_task-1, 3);
ASlower = zeros(2*period_of_task-1, 3);
PSupper = zeros(period_of_task-1,3);
PSlower = zeros(period_of_task-1,3);

%0 .. p => the aperiodic part is equal the computed arrival curve
for as_counter = 1:period_of_task
    ASupper(as_counter, 1) = as_counter-1;
    ASupper(as_counter, 2) = upperCurve(as_counter);
    ASlower(as_counter, 1) = as_counter-1;
    ASlower(as_counter, 2) = lowerCurve(as_counter);
  
end

%p .. 2p => choose the minimum/maximum for upper/lower curve
%           1) the arrival curve at position i
%           2) the sum of cache misses plus the arrival curve value at
%               position i-period.
for as_counter = period_of_task+1:2*period_of_task
    ASupper(as_counter, 1) = as_counter-1;
    value = max(upperCurve(as_counter), (sum(cache_misses_worst_case) + upperCurve(as_counter-period_of_task)));
    ASupper(as_counter, 2) = value;
    value_l = min(lowerCurve(as_counter), (sum(cache_misses_best_case) + lowerCurve(as_counter-period_of_task)));
    if(ASlower(as_counter-1,2) > value_l)
        value_l = ASlower(as_counter-1,2);
    end

    ASlower(as_counter, 1) = as_counter-1;
    ASlower(as_counter, 2) = value_l;
end

%compute the slopes accrodingly.
for as_counter = 1:2*period_of_task-1
 %  if(as_counter+1 < 2*period_of_task)
    ASlower(as_counter, 3) = ASlower(as_counter+1, 2) - ASlower(as_counter,2);
    ASupper(as_counter, 3) = ASupper(as_counter+1, 2) - ASupper(as_counter,2);
 %  else
 %   ASlower(as_counter, 3) = sum(cache_misses_best_case) + lowerCurve(as_counter+1-period_of_task) - ASlower(as_counter,2);
 %   ASupper(as_counter, 3) = sum(cache_misses_worst_case) + upperCurve(as_counter+1-period_of_task) - ASupper(as_counter,2);
 %  end
end

%the periodic part is k*sum(cache_misses_for_one_period)+the corresponding
%part of the lowerCurve/upperCurve
for ps_counter = 1:period_of_task
    %the values have to be offset, as the RTCcurve produces peaks otherwise
    offset_upper = upperCurve(1);
    offset_lower = lowerCurve(1);
    PSupper(ps_counter, 1) = ps_counter-1;
    PSupper(ps_counter, 2) = upperCurve(ps_counter);% - offset_upper;
    PSupper(ps_counter, 3) = upperCurve(ps_counter+1) - upperCurve(ps_counter);
    PSlower(ps_counter, 1) = ps_counter-1;
    PSlower(ps_counter, 2) = lowerCurve(ps_counter);% - offset_lower;
    PSlower(ps_counter, 3) = lowerCurve(ps_counter+1) - lowerCurve(ps_counter);
end

%the start values for the periodic parts, at the end of the aperiodic part
%start_value_upper = 2*(sum(cache_misses_worst_case)) + upperCurve(1);
%start_value_lower = 2*(sum(cache_misses_best_case)) + lowerCurve(1);

%the slope at the transition between aperiodic and periodic
ASlower(end,3 ) = PSlower(1,2);
%the slope of the last periodic element:
PSlower(end,3) = lowerCurve(1);

%create the arrival curves.
RTCcurve_upper = rtccurve(ASupper, PSupper, 2*period_of_task+1, ASupper(end,2), period_of_task);
RTCcurve_lower = rtccurve(ASlower, PSlower, 2*period_of_task+1, ASlower(end,2), period_of_task);

%PLOT
%createfigure([overall_upperCurve; overall_lowerCurve]');
%hold on
rtcplot(RTCcurve_upper, 'b', 1000, 'linewidth', 2);
hold on
rtcplot(RTCcurve_lower, 'r',1000, 'linewidth', 2);
end
