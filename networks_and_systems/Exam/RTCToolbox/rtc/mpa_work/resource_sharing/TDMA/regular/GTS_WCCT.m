function [wcrt_general_max_seq_triggered wcrt_general_superblocks_max_seq_triggered wcrt_general_superblocks_max general_triggered_unsched] = ...
    GTS_WCCT(tasks, length, duration, start_time, C, wcrt_general_superblocks_max, considered_task, period)

%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
general_triggered_unsched = 0;
wcrt_general_triggered = 0;
wcrt_general_max_seq_triggered = 0;
wcrt_general_superblocks_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_general_superblocks_max_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
current_time = 0;
hybrid_general_unsched = 0;
while(current_time <= lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6)
       
    for i = 1:size(tasks(considered_task).superblocks,2)


        execution_time_upper = tasks(considered_task).superblocks(i).execution_time_upper;
        accesses_read = tasks(considered_task).superblocks(i).accesses_upper(1);
        accesses_execute = tasks(considered_task).superblocks(i).accesses_upper(2);
        accesses_write = tasks(considered_task).superblocks(i).accesses_upper(3);

        
        if(i == 1)
            trigger_time = current_time;
        else
            trigger_time = wcrt_general_superblocks_max(i-1)+current_time;
        end
        %hybrid model, triggered blocks, sequential phases
       % wcrt_hybrid_seq_triggered = wct_ar(length, duration, start_time, accesses_read, trigger_time, C);
        wcrt_general_seq_triggered = wct_e( length, duration, start_time, accesses_execute+accesses_read+accesses_write, trigger_time, C, execution_time_upper);
        %wcrt_hybrid_seq_triggered = wct_ar(length, duration, start_time, accesses_write, wcrt_hybrid_seq_triggered, C);
        wcrt_general_superblocks_seq_triggered(i) = wcrt_general_seq_triggered;
        
                
        if(wcrt_general_seq_triggered - current_time > wcrt_general_superblocks_max(i)+10^-6)
            wcrt_general_superblocks_max(i) = wcrt_general_seq_triggered - current_time;
            general_triggered_unsched = 1;      
        end


    end
    
    wcrt_general_max_seq_triggered = max(wcrt_general_max_seq_triggered, wcrt_general_seq_triggered - current_time);
    wcrt_general_superblocks_max_seq_triggered = max(wcrt_general_superblocks_max_seq_triggered, wcrt_general_superblocks_seq_triggered-current_time);
    
    
    
    if lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6 > 1000
        break;
    end
    current_time = current_time + period;
    
%     if(general_triggered_unsched == 1)
%         wcrt_general_seq_triggered = 0;
%         current_time = 0;
%         wcrt_general_max_triggered = 0;
%         wcrt_general_superblocks_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         wcrt_general_superblocks_max_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         general_triggered_unsched = 0;     
%     end
    
end
end

