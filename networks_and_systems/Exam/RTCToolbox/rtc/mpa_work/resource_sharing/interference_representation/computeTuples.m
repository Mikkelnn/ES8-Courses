function tuples = computeTuples(distance, execution_time_lower, execution_time_upper, communication_time, cache_misses_best_case, cache_misses_worst_case, num_blocks, period_of_task)
%tuples = computeTuples(distance, execution_time_lower, execution_time_upper, communication_time, cache_misses_best_case, cache_misses_worst_case, num_blocks, period_of_task)
%
%This function compute the minimal/maximal number of cache misses for the
%worst and best case execution times of a task, that consists of multiple
%super blocks. The function is supposed to be executed iteratively for
%each distance. The distance corresponds to the number of subsequent
%superblocks that are considered for the analysis. 
%
%distance                   ... the number of subsequent blocks to be 
%                               considered (0 = each block is considered alone).
%exeution_time_lower        ... a vector storing the best case execution 
%                               time of each superblock
%exeution_time_upper        ... a vector storing the worst case execution 
%                               time of each superblock
%communication_time         ... the time penalty for one cache miss
%cache_misses_best_case     ... a vector storing the minimal number of 
%                               cache misses for the superblocks
%cache_misses_worst_case    ... a vector storing the maximal number of 
%                               cache misses for the superblocks
%num_blocks                 ... the absolut number of superblocks
%                               constituting the task
%period_of_task             ... the period of the task
%
%tuples ... the data structure representing the cache_miss_pattern
%tuples = 
%
%          best_et: [ 2*num_blocks values ]
%         worst_et: [ 2*num_blocks values ]
%    misses_bc_min: [ 2*num_blocks values ]
%    misses_wc_min: [ 2*num_blocks values ]
%    misses_bc_max: [ 2*num_blocks values ]
%    misses_wc_max: [ 2*num_blocks values ]
%
%   best_et         ... the best case execution times
%   worst_et        ... the worst case execution times
%   misses_bc_min   ... the minimal number of cache misses for BC exec.
%                       time
%   misses_wc_min   ... the minimal number of cache misses for WC exec.
%                       time
%   misses_bc_max   ... the maximal number of cache misses for BC exec.
%                       time
%   misses_wc_max   ... the maximal number of cache misses for WC exec.
%                       time
%
%See also computeCurves, arrivalCurvesTuples


    data_bestcase_u = zeros(1,(2*num_blocks));
    data_worstcase_u = zeros(1,(2*num_blocks));
    data_bestcase_l = zeros(1,(2*num_blocks));
    data_worstcase_l = zeros(1,(2*num_blocks));
    data_misses_bc_min = zeros(1,(2*num_blocks));
    data_misses_wc_min = zeros(1,(2*num_blocks));
    data_misses_bc_max = zeros(1,(2*num_blocks));
    data_misses_wc_max = zeros(1,(2*num_blocks));

    
    target_position = 1;
    %create a vector that states the actual index to access in the
    %data vectors. An easy way to get circular access
    access_vector = [1:num_blocks 1:num_blocks];
    
    %depending on the distance, up to 2*num_blocks offsets are 
    %possible. If the distance grows, some offsets are not feasible
    %anymore, as they would exceed over 2*num_blocks. 
    %Therefore stop at such an offset, that the values for the
    %distance under consideration can be accessed.
    for i = 1:length(access_vector)-distance
       if(distance+access_vector(i) > num_blocks ) 
           %overlap = 1;
           remaining_period_best_cases_u =  period_of_task - ( sum(execution_time_upper) + sum(cache_misses_best_case.*communication_time));
           remaining_period_worst_cases_u = period_of_task - ( sum(execution_time_upper) + sum(cache_misses_worst_case.*communication_time));
           remaining_period_best_cases_l =  period_of_task - ( sum(execution_time_lower) + sum(cache_misses_best_case.*communication_time));
           remaining_period_worst_cases_l = period_of_task - ( sum(execution_time_lower) + sum(cache_misses_worst_case.*communication_time));
       else
           remaining_period_best_cases_u =  0;
           remaining_period_worst_cases_u = 0;
           remaining_period_best_cases_l =  0;
           remaining_period_worst_cases_l = 0;
       end   
       
       
       %sum up the execution times and cache miss times as well as
       %the actual cache misses.
       for sum_counter=0:distance
         best_value_u = 0;
         worst_value_u = 0;
         best_value_l = 0;
         worst_value_l = 0;
         index = access_vector(i+sum_counter);
           
         %leave out the execution time of the first and last block
         %those times are not necessary for the cache misses to happen,
         %the exectution time of the first superblock can be before the
         %time window and the one for the last superblock can be after the
         %time window under consideration.
         if(sum_counter ~= 0 && sum_counter ~= distance)
           best_value_u = execution_time_upper(index);
           worst_value_u = execution_time_upper(index);
           best_value_l = execution_time_lower(index);
           worst_value_l = execution_time_lower(index);
         end
         %the cache miss time of the last element of subsequent
         %super-blocks does not need to be considered. the assumption is
         %that cache misses occur at the beginning of the cache miss
         %related computation time.
         if(sum_counter ~= distance)
           best_value_u = best_value_u + (cache_misses_best_case(index) * communication_time);
           worst_value_u = worst_value_u + (cache_misses_worst_case(index) * communication_time);
           best_value_l = best_value_l + (cache_misses_best_case(index) * communication_time);
           worst_value_l = worst_value_l + (cache_misses_worst_case(index) * communication_time);
         end
               
         misses_value_bc_min = cache_misses_best_case(index);
         misses_value_wc_min = cache_misses_worst_case(index);
         misses_value_bc_max = cache_misses_best_case(index);
         misses_value_wc_max = cache_misses_worst_case(index);
         
         %for the lower curve computation, leave out the last
         %block of cache misses. they don't have to occur, as
         %the time for their execution isn't part of the time-dealine.
         %the don't have to occur for the best and neither for the worst
         %case execution time.
         if(sum_counter == distance)
             misses_value_bc_min = 0;
             misses_value_wc_min = 0;
         end
         
         data_bestcase_u(target_position) = data_bestcase_u(target_position) + best_value_u;
         data_worstcase_u(target_position) = data_worstcase_u(target_position) + worst_value_u;
         data_bestcase_l(target_position) = data_bestcase_l(target_position) + best_value_l;
         data_worstcase_l(target_position) = data_worstcase_l(target_position) + worst_value_l;
         data_misses_bc_min(target_position) = data_misses_bc_min(target_position) + misses_value_bc_min;
         data_misses_wc_min(target_position) = data_misses_wc_min(target_position) + misses_value_wc_min; 
         data_misses_bc_max(target_position) = data_misses_bc_max(target_position) + misses_value_bc_max;
         data_misses_wc_max(target_position) = data_misses_wc_max(target_position) + misses_value_wc_max; 
       end
       %if we have the gap to the next period, add it up here.
       data_bestcase_u(target_position) = data_bestcase_u(target_position) + remaining_period_best_cases_u;
       data_worstcase_u(target_position) = data_worstcase_u(target_position) + remaining_period_worst_cases_u;
       data_bestcase_l(target_position) = data_bestcase_l(target_position) + remaining_period_best_cases_l;
       data_worstcase_l(target_position) = data_worstcase_l(target_position) + remaining_period_worst_cases_l;
       target_position = target_position +1;      
        
    end
    
    %write the data structure
    tuples.best_et_u = data_bestcase_u;
    tuples.worst_et_u = data_worstcase_u;
    tuples.best_et_l = data_bestcase_l;
    tuples.worst_et_l = data_worstcase_l;
    tuples.misses_bc_min = data_misses_bc_min;
    tuples.misses_wc_min = data_misses_wc_min;
    tuples.misses_bc_max = data_misses_bc_max;
    tuples.misses_wc_max = data_misses_wc_max;


end