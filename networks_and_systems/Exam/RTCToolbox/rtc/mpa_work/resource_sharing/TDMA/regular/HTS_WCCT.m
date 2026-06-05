function [wcrt_hybrid_max_seq_triggered wcrt_hybrid_superblocks_max_seq_triggered  wcrt_hybrid_superblocks_max, hybrid_triggered_unsched] = ...
    HTS_WCCT(tasks, length, duration, start_time, C,  wcrt_hybrid_superblocks_max, considered_task, period)

wcrt_hybrid_triggered = 0;
wcrt_hybrid_max_seq_triggered = 0;
wcrt_hybrid_superblocks_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_max_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
current_time = 0;
hybrid_triggered_unsched = 0;
while(current_time <= lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6)
       
    for i = 1:size(tasks(considered_task).superblocks,2)


        execution_time_upper = tasks(considered_task).superblocks(i).execution_time_upper;
        accesses_read = tasks(considered_task).superblocks(i).accesses_upper(1);
        accesses_execute = tasks(considered_task).superblocks(i).accesses_upper(2);
        accesses_write = tasks(considered_task).superblocks(i).accesses_upper(3);

        
        if(i == 1)
            trigger_time = current_time;
        else
            trigger_time = wcrt_hybrid_superblocks_max(i-1)+current_time;
        end
        %hybrid model, triggered blocks, sequential phases
        wcrt_hybrid_seq_triggered = wct_ar(length, duration, start_time, accesses_read, trigger_time, C);
        wcrt_hybrid_seq_triggered = wct_e( length, duration, start_time, accesses_execute, wcrt_hybrid_seq_triggered, C, execution_time_upper);
        wcrt_hybrid_seq_triggered = wct_ar(length, duration, start_time, accesses_write, wcrt_hybrid_seq_triggered, C);
        wcrt_hybrid_superblocks_seq_triggered(i) = wcrt_hybrid_seq_triggered;
        
                
        if(wcrt_hybrid_seq_triggered - current_time > wcrt_hybrid_superblocks_max(i)+10^-6)
            wcrt_hybrid_superblocks_max(i) = wcrt_hybrid_seq_triggered - current_time;
            hybrid_triggered_unsched = 1;      
        end


    end
    
    wcrt_hybrid_max_seq_triggered = max(wcrt_hybrid_max_seq_triggered, wcrt_hybrid_seq_triggered - current_time);
    wcrt_hybrid_superblocks_max_seq_triggered = max(wcrt_hybrid_superblocks_max_seq_triggered, wcrt_hybrid_superblocks_seq_triggered-current_time);
       
    
    if lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6 > 1000
        break;
    end
    current_time = current_time + period;
    
%     if(hybrid_triggered_unsched == 1)
%         wcrt_hybrid_seq_triggered = 0;
%         current_time = 0;
%         wcrt_hybrid_max_triggered = 0;
%         wcrt_hybrid_superblocks_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         wcrt_hybrid_superblocks_max_seq_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         hybrid_triggered_unsched = 0;     
%     end
    
end

end