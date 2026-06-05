
function [RTCcurve_upper] = interference_single_periodic_task(config_file, output, delta, debug)
%INTERFERENCE OF A SINGLE PERIODIC TASK ON A SHARED RESOURCE
%The function computes the interference of a task composed of super-blocks
%on a shared resource in MPA-RTC arrival curve representation. 
%Super-blocks are defined with their minimal and maximal execution
%time and their minimal and maximal number of access to the shared
%resource.
%The resulting interference is given as an upper and lower arrival
%curve, representing the maximal and minimal interference respectively.
%
%ATTENTION: This script expects the RTC Toolbox to be available. You can
%download the RTC Toolbox at http://www.mpa.ethz.ch/.
%
%[RTCcurve_lower, RTCcurve_upper] = interference_single_periodic_task(config_file)
%
%RTCcurve_upper         ... the upper arrival curve
%   RTCcurve_upper(1,:) ... the indices representing the time window
%   RTCcurve_upper(2,:) ... the value representing the access to the shared
%                           resource
%   RTCcurve_upper(3,:) ... the slope of the curve for the current section
%config_file            ... the config file handle that stores the 
%                           superblock information. 
%                           config_file = load('\path\to\file');
%output                 ... 1, if the resulting arrival curves should be
%                           plotted, 0 otherwise
%delta                  ... if output = 1, delta gives the bound of the
%                           x-axis of the plot, e.g., delta = 100 plots the
%                           arrival curves for time windows 0...100.
%debug                  ... 1, if the tuples that are used to compute the
%                           arrival curves should be plotted, 0 otherwise
%
%As an example config_file, copy the following lines to a file and open
%this file in MATLAB as: config_file = load('\path\to\file');
%%------------ BEGIN COPY AFTER THIS LINE ----------------------------%%
% %number of superblocks
% %period
% %number of cache misses - best case
% %number of cache misses - worst case
% %communication time per cache miss
% %execution time - best case
% %execution time - worst case
% 10
% 285
% 7 2 9 3 11 2 5 5 12 13
% 10 12 9 10 14 4 6 12 13 18
% 2
% 8 9 5 4 7 6 3 1 2 5
% 10 30 9 10 15 9 4 2 4 5
%%------------ END COPY BEFORE THIS LINE -----------------------------%%
%
%See also computeCurves, computeTuples, rtccurve


    [A D H] = importdata(config_file, ' ',7);
    number_of_super_blocks = A.data(1);
    period_of_task = A.data(2);
    cache_misses_best_case = A.data(3:3+number_of_super_blocks-1);
    cache_misses_worst_case = A.data(3+number_of_super_blocks:3+2*number_of_super_blocks-1);
    communication_time_per_miss = A.data(3+2*number_of_super_blocks);
    execution_time_worst = A.data(3+2*number_of_super_blocks+1:3+3*number_of_super_blocks);
    execution_time_best = A.data(3+3*number_of_super_blocks+1:3+4*number_of_super_blocks);

    upperCurve = zeros(1, 2*period_of_task);
    lowerCurve = Inf*ones(1, 2*period_of_task);
    plotUpper = zeros(1, 2*period_of_task);
    plotLower = Inf*ones(1, 2*period_of_task);
    oldUpper = zeros(1, 2*period_of_task);
    oldLower = Inf*ones(1, 2*period_of_task);

    %loop over all possible distances. The distance is the
    %number of subsequent superblocks considered.
    %distance = 0 means means, that each superblocks values
    %are considered for the tuples,
    %distance = num_blocks means, that the sum over all superblocks
    %is considered for constructing the tuples
    if(debug == 1)
        figure;
        hold on;
        marker_array = ['o', 's', '<', '>', 'v', '^', '+', '.', 'p', 'h', '*', '-'];
        while(size(marker_array,2) <= 2*number_of_super_blocks)
            marker_array = [marker_array, marker_array];
        end
    end
    for distance = 0:2*number_of_super_blocks-1 
        tuples = computeTuples(distance, execution_time_best, execution_time_worst, communication_time_per_miss, cache_misses_best_case, cache_misses_worst_case, number_of_super_blocks, period_of_task);
        %for each set of tuples, update the curves
        [upperCurve lowerCurve] = computeCurves(tuples, upperCurve, lowerCurve, distance);
        changesUpper = upperCurve - oldUpper;
        changesLower = lowerCurve - oldLower;
        indexesUpperChanges = find(changesUpper ~= 0);
        indexesLowerChanges = find(changesLower ~= 0);
        
        plotUpper = zeros(1, 2*period_of_task);
        plotLower = Inf*ones(1, 2*period_of_task);
        
        plotUpper(indexesUpperChanges) = upperCurve(indexesUpperChanges);
        plotLower(indexesLowerChanges) = lowerCurve(indexesLowerChanges);
                       
        oldUpper = upperCurve;
        oldLower = lowerCurve;
        
        plotUpper(find(plotUpper == 0)) = Inf;
        
        if (debug == 1)
            command_green = sprintf('%s%s', 'g', marker_array(distance+1));
            command_red = sprintf('%s%s', 'm', marker_array(distance+1));
            label = sprintf('%d %s', distance+1, 'superblocks');
            %note that the curves need to be shifted by -1, so that they are
            %aligned with the arrival curve later on (rtcplot produces curves
            %that start at index 0, which is uncommon in matlab).
            plot([0:length(plotUpper)-1], plotUpper, command_green, 'DisplayName', label);
            hold on;
            plot([0:length(plotLower)-1], plotLower, command_red, 'DisplayName', label);
            hold on;
        end

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
        ASlower(as_counter, 3) = ASlower(as_counter+1, 2) - ASlower(as_counter,2);
        ASupper(as_counter, 3) = ASupper(as_counter+1, 2) - ASupper(as_counter,2);
    end

    %the periodic part is k*sum(cache_misses_for_one_period)+the corresponding
    %part of the lowerCurve/upperCurve
    for ps_counter = 1:period_of_task
        PSupper(ps_counter, 1) = ps_counter-1;
        PSupper(ps_counter, 2) = upperCurve(ps_counter);
        PSupper(ps_counter, 3) = upperCurve(ps_counter+1) - upperCurve(ps_counter);
        PSlower(ps_counter, 1) = ps_counter-1;
        PSlower(ps_counter, 2) = lowerCurve(ps_counter);
        PSlower(ps_counter, 3) = lowerCurve(ps_counter+1) - lowerCurve(ps_counter);
    end


    %the slope at the transition between aperiodic and periodic
    ASlower(end,3 ) = PSlower(1,2);
    %the slope of the last periodic element:
    PSlower(end,3) = lowerCurve(1);

    %create the arrival curves.
    RTCcurve_upper = rtccurve(ASupper, PSupper, 2*period_of_task+1, ASupper(end,2), period_of_task);
%    RTCcurve_lower = rtccurve(ASlower, PSlower, 2*period_of_task+1, ASlower(end,2), period_of_task);

    
    %plot MPA curves
    if(output == 1)
        %figure;
        rtcplot(RTCcurve_upper, 'b', delta, 'linewidth', 2, 'DisplayName', 'upper arrival curve \hat{\alpha}^u');
%        hold on
%        rtcplot(RTCcurve_lower, 'r', delta, 'linewidth', 2,'DisplayName', 'lower arrival curve \hat{\alpha}^l');
    end
    

    
end
