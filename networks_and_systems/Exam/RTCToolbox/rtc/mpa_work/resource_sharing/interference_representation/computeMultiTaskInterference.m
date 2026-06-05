function RTCupperCurve = computeMultiTaskInterference(config_file, output, delta)
%RTCupperCurve = computeMultiTaskInterference(config_file, output)
%
%This function computes an RTC uppper curve, from a configuration file
%reprsenting a set of tasks executing on a processing element in a static
%periodic schedule, i.e. in a time-wheel.
%The curve represent the access pattern of the tasks on one processing unit
%to a shared resource (e.g., cache, I/O).
%
%'config_file'   ...  is the configuration file explained below.
%                     with config_file = load('\path\to\file\file.txt');
%'output'        ...  set to 1 to plot the resulting curve 
%'delta'         ...  the length of the x-axis, if output=1
%The configuration file has the following structure:
%
%
%-----------BEGIN FILE - COPY FROM HERE ------------------------------%
%%Number of tasks
%%the tasks time-slots (e.g. task 1 has 10 time units, followed by task 2
%%with 8 time units).
%%the number of super-blocks in each task (e.g. task 1 has 2 super-blocks,
%%task 2 has 1 super-block)
%%for each task 4 lines, representing the following data in that order:
%	%the min computational time needed for each super-block
%	%
%	%the max computational time needed for each super-block
%	%
%	%the min amount of cache-misses that can happen for each super block
%	%
%	%the max amount of cache-misses that can happen for each super block
%2
%10 8
%2 1
%3 2
%5 3
%1 1
%2 2
%4 
%7
%1
%2
%-----------END FILE - COPY UNTIL HERE ------------------------------%
%
%
%NOTE: The RTC toolbox is required for this function to work. Please
%download at www.mpa.ethz.ch.


%process the input file data
raw_data = importdata(config_file, ' ', 13);
number_of_tasks = raw_data.data(1);
time_wheel = raw_data.data(2:2+number_of_tasks-1);
offset = 2+number_of_tasks;
number_of_super_blocks = raw_data.data(offset:offset+number_of_tasks-1);
offset = offset +number_of_tasks;

%prepare the data structures:
%a) a vector containing the min number of cache-misses, with a zero block
%representing the slot at the end of a task (the gap to the instantiation
%time of the next task)
%b) a vector containing the max number of cache-misses, similarly
%c) a vector containing the min execution time of each super-block in the
%task set. The gap between the end of one task and the beginning of a new
%task is represented by a virtual super-block, issuing no cache-miss but
%with an execution time that computes as: gap = time_slot_of_task -
%sum(min_execution_time_of_all_superblocks).
%d) a vector containing the max execution time of each super-block,
%similarly
execution_time_lower = zeros(1, (sum(number_of_super_blocks)+number_of_tasks));
execution_time_upper = zeros(1, (sum(number_of_super_blocks)+number_of_tasks));
cache_miss_lower = zeros(1, (sum(number_of_super_blocks)+number_of_tasks));
cache_miss_upper = zeros(1, (sum(number_of_super_blocks)+number_of_tasks));


vector_offset = 1;
for i=1:number_of_tasks
    %the number of superblocks in the current tasks
    curr_sb = number_of_super_blocks(i);
    
    %lower bound on the execution time 
    execution_time_lower(vector_offset:vector_offset-1+curr_sb) = raw_data.data(offset:offset+curr_sb-1)';
    execution_time_lower(vector_offset+curr_sb) = 99;
    offset = offset+curr_sb;
    
    %upper bound on the execution time
    execution_time_upper(vector_offset:vector_offset-1+curr_sb) = raw_data.data(offset:offset+curr_sb-1)';
    execution_time_upper(vector_offset+curr_sb) = 99;
    offset = offset+curr_sb;
    
    %lower bound on the number of cache-misses
    cache_miss_lower(vector_offset:vector_offset-1+curr_sb) = raw_data.data(offset:offset+curr_sb-1)';
    cache_miss_lower(vector_offset+curr_sb) = 0;
    offset = offset+curr_sb;
    
    %upper bound on the number of cache-misses
    cache_miss_upper(vector_offset:vector_offset-1+curr_sb) = raw_data.data(offset:offset+curr_sb-1)';
    cache_miss_upper(vector_offset+curr_sb) = 0;
    offset = offset+curr_sb;
    
    %next tasks offset
    vector_offset = vector_offset + curr_sb + 1;
end

%double the vectors, as we consider two periods to compute the arrival
%curves
execution_time_lower = [ execution_time_lower execution_time_lower];
execution_time_upper = [ execution_time_upper execution_time_upper];
cache_miss_lower = [ cache_miss_lower cache_miss_lower];
cache_miss_upper = [ cache_miss_upper cache_miss_upper];
size_tuples = (2*(sum(number_of_super_blocks)+number_of_tasks))*(2*(sum(number_of_super_blocks)+number_of_tasks));
max_number_of_cache_misses = zeros(1, size_tuples);
max_execution_time = zeros(1, size_tuples);

%%start the computation of the cache-access pattern
counter = 1;
for width_counter = 0:2*(sum(number_of_super_blocks)+number_of_tasks-1)
    for start_index = 1:2*(sum(number_of_super_blocks)+number_of_tasks)
          
          %compute for every possible width, for up to two periods of the
          %time wheel. Break the loop if the windows 
          width =  width_counter;
          if (start_index + width >= 2*(sum(number_of_super_blocks)+number_of_tasks))
              width = 2*(sum(number_of_super_blocks)+number_of_tasks) -  start_index;
              continue;
          end
          
          %to get the highest number of possible cache-misses in the
          %shortest amount of time:
          %a) start with the first cache-miss block as late as possible
          %(i.e. after the computational block of the first super-block)
          %b) stop with the last cache-miss block as soon as possible (i.e.
          %the last cache-miss block is issued before its corresponding
          %computational block.
          %c) all super-block between the first and the last have no
          %restriction on their order 
          %d) once all superblocks for a task are considered, consider the
          %gap (for worstcase: timeslot - sum(upper_exec_time), for
          %bestcase: timeslot - sum(lower_exec_time)
          
          max_number_of_cache_misses(counter) = sum(cache_miss_upper(start_index:start_index+width));
          
          times = execution_time_lower(start_index+1:start_index+width-1);
          %determine the number of involved tasks
          number_of_considered_tasks = length(find(times == 99)) +1;
          %if(execution_time_lower(start_index+width) == 99)
          %    number_of_considered_tasks = number_of_considered_tasks - 1;
          %end
          
          %determine the first task
          position_of_task = find(execution_time_lower(1:start_index) == 99);
          task_number = length(position_of_task) + 1;
          if(task_number > number_of_tasks)
              task_number = task_number - number_of_tasks;
          end
          %if(execution_time_lower(start_index)  == 99)
          %    task_number = task_number - 1;
          %end
          
          %compute the first super-block in the task, considered for the
          %time window. Then compute the gap if there are more then 1 tasks
          %involved.
          gap = 0;
          if(number_of_considered_tasks > 1)
            %find the gaps
            gap_indices = find(times == 99);
            %determine the super-block inside a task, at which the
            %computation of the current time-window begins
            number_of_sb_before_window = number_of_super_blocks(task_number)+1 - gap_indices(1);
            end_of_task_index = gap_indices(1)-1;
            %compute the gap:
            %- in order to compute a worst-case cache-access, we have to find
            %the highest number of cache-misses, for the smallest
            %time-windows.
            %- time-windows are computed by computing the time a number of
            %super-blocks need to execute (the lower bound).
            %- time-windows can span over several tasks (depending on their
            %width) and their start-index is shifted over all possible
            %positions
            %- as a consequence it is possible that a window for computing
            %such a sequence starts at some point in task 1 and continues
            %up to some super-block in task 2. In this case, those
            %super-blocks in task 1 that are not inside the window have to
            %be considered with their maximum computational time, while
            %those super-blocks inside the window have to be considered
            %with their minimum computational time. This reduces the gap at
            %the end of the task, which in this case is part of the
            %time-window, and therefore results in a higher number of
            %cache-misses for a shorter time window, thus representing the
            %worst case.
            gap = time_wheel(task_number) - sum(execution_time_upper(start_index-number_of_sb_before_window+1:start_index)) ...
                - sum(execution_time_lower(start_index+1:start_index+end_of_task_index));
            %compensate for the 99 values in the input vector
            max_execution_time(counter) = max_execution_time(counter) - ((number_of_considered_tasks-1)*99);
            
            %add the time for the cache misses
            
            
            gap_indices(1) = [];
          end
          
          %compute other gaps, if there are more than 2 tasks involved:
          intermediate_gaps = zeros(1, number_of_considered_tasks);
          if( (number_of_considered_tasks > 2))

            loop = 0;
            %current_window_size = width;
            for gap_counter = 1:number_of_considered_tasks-1
                if (~isempty(gap_indices) )
                %leave the first gap - it's considered before - start with
                %the second task
                current_task = task_number + gap_counter;
                if current_task > number_of_tasks
                    current_task = current_task - number_of_tasks;
                    loop = loop + 1;
                end

                %compute the super-blocks belonging to this task
                offset_super_blocks = sum(number_of_super_blocks(1:current_task-1));
                offset_super_blocks = offset_super_blocks + current_task-1;

                intermediate_gaps(gap_counter) = time_wheel(current_task) - ...
                    sum(execution_time_lower(offset_super_blocks+1:offset_super_blocks+1+number_of_super_blocks(current_task)-1));
                
                %no need to add time for cache misses for the intermediate
                %gaps, since everything has to sum up to the complete
                %time wheel anyways.
                
                gap_indices(1) = [];
                end
            end
          end
          
          
          max_execution_time(counter) = max_execution_time(counter) + sum(execution_time_lower(start_index+1:start_index+width-1)) ...
              + gap + sum(intermediate_gaps);
          %add the time of the cache misses
          
          %compute the minimal as well
          
          
          counter = counter + 1;
          

    end
    
end

%create a continuous curve where the index equals time, in order to process it with MPA
curve = zeros(1,2*sum(time_wheel));
for curve_counter = 1:length(curve)
    [value index] = find(max_execution_time == curve_counter-1);
    if(~isempty(index))
        curve(curve_counter) = max(max_number_of_cache_misses(index));
    end
    if(curve_counter ~= 1)
        if(curve(curve_counter-1) > curve(curve_counter))
            curve(curve_counter) = curve(curve_counter-1);
        end
    end
end

RTCupperCurve = generateRTCCurve(curve, sum(time_wheel), cache_miss_upper(1:length(cache_miss_upper)/2));
if output == 1 
    rtcplot(RTCupperCurve, 'b', delta, 'linewidth', 2);
end
end
