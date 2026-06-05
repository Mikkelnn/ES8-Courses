
function [wcrt_hybrid_superblocks_max_trig_triggered] = ...
    HTT_WCCT(tasks, length, duration, start_time, C, wcrt_hybrid_superblocks_max_acq_phases, wcrt_hybrid_superblocks_max_exec_phases, wcrt_hybrid_superblocks_max_rep_phases, considered_task, period)

current_time = 0;
wcrt_hybrid_max_trig_triggered = 0;
wcrt_hybrid_superblocks_trig_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
wcrt_hybrid_superblocks_max_trig_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
trig_triggered_unsched = 0;

while(current_time <= lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6)
    
    

    for i = 1:size(tasks(considered_task).superblocks,2)    
        
        

        execution_time_upper = tasks(considered_task).superblocks(i).execution_time_upper;
        accesses_read = tasks(considered_task).superblocks(i).accesses_upper(1);
        accesses_execute = tasks(considered_task).superblocks(i).accesses_upper(2);
        accesses_write = tasks(considered_task).superblocks(i).accesses_upper(3);
        
        
        if(i == 1)
            trigger_time = current_time;
        else
            trigger_time = wcrt_hybrid_superblocks_max_rep_phases(i-1)+current_time;
            %wcrt_hybrid_superblocks_max_rep_phases(i-1);
            %wcrt_hybrid_superblocks_max(i-1);
            
        end
        %hybrid model, triggered blocks, triggered phases
        wcrt_hybrid_trig_triggered = wct_ar(length, duration, start_time, accesses_read, trigger_time, C);
        if(wcrt_hybrid_trig_triggered - current_time > wcrt_hybrid_superblocks_max_acq_phases(i)+10^-6)
            trig_triggered_unsched = 1;
            wcrt_hybrid_superblocks_max_acq_phases(i) = wcrt_hybrid_trig_triggered - current_time;
        end
        
        wcrt_hybrid_trig_triggered = wct_e( length, duration, start_time, accesses_execute,  wcrt_hybrid_superblocks_max_acq_phases(i)+current_time , C, execution_time_upper);
        if(wcrt_hybrid_trig_triggered - current_time > wcrt_hybrid_superblocks_max_exec_phases(i)+10^-6)
            trig_triggered_unsched = 1;
            wcrt_hybrid_superblocks_max_exec_phases(i) = wcrt_hybrid_trig_triggered - current_time;
        end
        
        wcrt_hybrid_trig_triggered = wct_ar(length, duration, start_time, accesses_write, wcrt_hybrid_superblocks_max_exec_phases(i)+current_time, C);
        if(wcrt_hybrid_trig_triggered - current_time > wcrt_hybrid_superblocks_max_rep_phases(i)+10^-6)
            trig_triggered_unsched = 1;
            wcrt_hybrid_superblocks_max_rep_phases(i) = wcrt_hybrid_trig_triggered - current_time;
        end

        wcrt_hybrid_superblocks_trig_triggered(i) = wcrt_hybrid_trig_triggered;
    
        

    
    end
    
    wcrt_hybrid_max_trig_triggered = max(wcrt_hybrid_max_trig_triggered, wcrt_hybrid_trig_triggered - current_time);
    %if(total_trig_triggered_unsched == 1)
     %   wcrt_hybrid_max_trig_triggered = wcrt_hybrid_max_trig_triggered + penalty;
    %    total_trig_triggered_unsched = 0;
    %end
    wcrt_hybrid_superblocks_max_trig_triggered = max(wcrt_hybrid_superblocks_max_trig_triggered, wcrt_hybrid_superblocks_trig_triggered-current_time);
    
    
%     if(trig_triggered_unsched == 1)
%         current_time = 0;
%         total_trig_triggered_unsched = 0;
%         wcrt_hybrid_max_trig_triggered = 0;
%         wcrt_hybrid_superblocks_trig_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         wcrt_hybrid_superblocks_max_trig_triggered = zeros(1, size(tasks(considered_task).superblocks,2));
%         trig_triggered_unsched = 0;
%         
%     end
    if lcm(ceil(10^6*length), 10^6* tasks(considered_task).period)/10^6> 1000
        break;
    end
    current_time = current_time + period;
    
end
wcrt_hybrid_superblocks_max_trig_triggered = max(wcrt_hybrid_superblocks_max_trig_triggered);
end