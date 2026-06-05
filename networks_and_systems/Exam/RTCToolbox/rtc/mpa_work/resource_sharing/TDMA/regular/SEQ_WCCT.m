function [ wcrt_general_max ...
    wcrt_hybrid_max ...
    wcrt_dedicated_max ...
    wcrt_dedicated_superblocks_max ...
    wcrt_hybrid_superblocks_max ...
    wcrt_general_superblocks_max ...
    wcrt_hybrid_superblocks_max_acq_phases ...
    wcrt_hybrid_superblocks_max_exec_phases ...
    wcrt_hybrid_superblocks_max_rep_phases ] = ...
    SEQ_WCCT(tasks, length, duration, start_time, C, considered_task, period)

wcrt_hybrid = 0;
wcrt_dedicated = 0;
wcrt_general = 0;
current_time = 0;

wcrt_hybrid_max = 0;
wcrt_dedicated_max = 0;
wcrt_general_max = 0;

wcrt_hybrid_superblocks = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_dedicated_superblocks = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_general_superblocks = zeros(1, size(tasks(considered_task).superblocks,2));

wcrt_hybrid_superblocks_max = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_dedicated_superblocks_max = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_general_superblocks_max = zeros(1, size(tasks(considered_task).superblocks,2));

wcrt_hybrid_superblocks_max_acq_phases = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_max_exec_phases = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_max_rep_phases = zeros(1, size(tasks(considered_task).superblocks,2));

wcrt_hybrid_superblocks_acq_phases = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_exec_phases = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_rep_phases = zeros(1, size(tasks(considered_task).superblocks,2));


while(current_time <= lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6)
    
    wcrt_dedicated = current_time;
    wcrt_hybrid = current_time;
    wcrt_general = current_time;

       
    for i = 1:size(tasks(considered_task).superblocks,2)


        execution_time_upper = tasks(considered_task).superblocks(i).execution_time_upper;
        accesses_read = tasks(considered_task).superblocks(i).accesses_upper(1);
        accesses_execute = tasks(considered_task).superblocks(i).accesses_upper(2);
        accesses_write = tasks(considered_task).superblocks(i).accesses_upper(3);

        %dedicated model
        wcrt_dedicated = wct_ar(length, duration, start_time, accesses_read, wcrt_dedicated, C);
        wcrt_dedicated = wcrt_dedicated + execution_time_upper;
        wcrt_dedicated = wct_ar(length, duration, start_time, accesses_write+accesses_execute, wcrt_dedicated, C);
        wcrt_dedicated_superblocks(i) = wcrt_dedicated;

        %hybrid model
        wcrt_hybrid = wct_ar(length, duration, start_time, accesses_read, wcrt_hybrid, C);
        wcrt_hyb_acquisition = wcrt_hybrid;
        wcrt_hybrid = wct_e( length, duration, start_time, accesses_execute, wcrt_hybrid, C, execution_time_upper);
        wcrt_hyb_execution = wcrt_hybrid;
        wcrt_hybrid = wct_ar(length, duration, start_time, accesses_write, wcrt_hybrid, C);
        wcrt_hybrid_replication = wcrt_hybrid;
        wcrt_hybrid_superblocks(i) = wcrt_hybrid;
        
        %also store the wcrt of each phase, for later use in the total time
        %triggered model
        wcrt_hybrid_superblocks_acq_phases(i) = wcrt_hyb_acquisition;
        wcrt_hybrid_superblocks_exec_phases(i) = wcrt_hyb_execution;
        wcrt_hybrid_superblocks_rep_phases(i) =  wcrt_hybrid_replication;

        

        %general_model
        wcrt_general = wct_e( length, duration, start_time, accesses_execute+accesses_write+accesses_read, wcrt_general, C, execution_time_upper);
        wcrt_general_superblocks(i) = wcrt_general;

    end
    
    wcrt_general_max = max(wcrt_general_max, wcrt_general - current_time);
    wcrt_hybrid_max = max(wcrt_hybrid_max, wcrt_hybrid - current_time);
    wcrt_dedicated_max = max(wcrt_dedicated_max, wcrt_dedicated - current_time);
    
    wcrt_dedicated_superblocks_max = max(wcrt_dedicated_superblocks_max, wcrt_dedicated_superblocks-current_time);
    wcrt_hybrid_superblocks_max = max(wcrt_hybrid_superblocks_max, wcrt_hybrid_superblocks-current_time);
    wcrt_general_superblocks_max = max(wcrt_general_superblocks_max, wcrt_general_superblocks-current_time);
    
    %the hybrid models phases max. times.
    wcrt_hybrid_superblocks_max_acq_phases = max(wcrt_hybrid_superblocks_max_acq_phases,wcrt_hybrid_superblocks_acq_phases-current_time);
    wcrt_hybrid_superblocks_max_exec_phases = max(wcrt_hybrid_superblocks_max_exec_phases,wcrt_hybrid_superblocks_exec_phases-current_time);
    wcrt_hybrid_superblocks_max_rep_phases = max(wcrt_hybrid_superblocks_max_rep_phases, wcrt_hybrid_superblocks_rep_phases-current_time);
    
  
       
    
    if lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6 > 1000
        break;
    end
    current_time = current_time + period;
    
end