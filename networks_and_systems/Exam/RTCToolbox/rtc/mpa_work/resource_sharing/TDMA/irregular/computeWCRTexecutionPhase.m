function wcrt = computeWCRTexecutionPhase(tdma_arbiter, tdma_time_step, time, p, C, accesses, execution)



tdma_length = size(tdma_arbiter,2) * tdma_time_step;
cycles_length = floor(time / tdma_length);
rel_time = time - cycles_length*tdma_length;

current_slot = ceil(rel_time / tdma_time_step); %if the time is exactly at a slot border - it's the next one;
if( uint32(rel_time / tdma_time_step) == current_slot)
    current_slot = current_slot +1;
end
if(current_slot == 0)
    current_slot = 1;
end
if(current_slot > size(tdma_arbiter,2))
    current_slot = 1;
    %cycles_length = cycles_length + 1;
end
starting_time_current_time_slot = (current_slot-1) * tdma_time_step + cycles_length*tdma_length;


if(accesses == 0)
    wcrt = time+execution;
    return;
    
else
    %a = tdma_arbiter(current_slot);
    
    if(tdma_arbiter(current_slot) ~= p)
        %get starting point of next slot for p     
        [offset next_slot_starting_time]= getNextTimeSlotForP(tdma_arbiter, p, current_slot, tdma_time_step);
        next_starting_slot = starting_time_current_time_slot+next_slot_starting_time;
        wcrt1 = computeWCRTexecutionPhase(tdma_arbiter, tdma_time_step, next_starting_slot+C, p, C, accesses-1, execution);
        
        if( (execution - next_starting_slot + time) > 0)
            %if(~done)
                wcrt2 = computeWCRTexecutionPhase(tdma_arbiter, tdma_time_step, next_starting_slot, p, C, accesses, execution - next_starting_slot+time);
                wcrt1 = max(wcrt1, wcrt2);
                %return;
            %end
        end
        wcrt = wcrt1;
        return;
    end
    
    [offset next_slot_starting_time] = getNextTimeSlot(tdma_arbiter, current_slot, tdma_time_step);
    next_starting_slot = starting_time_current_time_slot+next_slot_starting_time;
    
    if((next_starting_slot - C) - time >= execution)
        wcrt = computeWCRTdedicatedPhase(tdma_arbiter, tdma_time_step, time+execution, p, C, accesses);
        return;
    end
    
    execution_left = execution - max(0, (next_starting_slot - C) - time);
    accesses_left = accesses - 1;
    
    %tdmad = [tdma_arbiter tdma_arbiter];
    indices_slots_assigned_to_p = find(tdma_arbiter == p);
    number_indices_assigned_to_p = size(indices_slots_assigned_to_p,2);
    %find the number of incontinuous section = this is the number of slots
    %then;
%     slot_count=0;
%     for y = 1:number_indices_assigned_to_p
%         if(y ~= 1)
%             if (indices_slots_assigned_to_p(y) ~= indices_slots_assigned_to_p(y-1)+1)
%                 slot_count = slot_count +1;
%             end
%         end
%     end
     slot_count=0;
     for y = 1:number_indices_assigned_to_p
         if(y == 1)
             slot_count = slot_count+ 1;
         else
             if (indices_slots_assigned_to_p(y) ~= indices_slots_assigned_to_p(y-1)+1)
                 slot_count = slot_count +1;
             end
         end
     end
    
    %\Psi^*
    Psi = slot_count;
    
    time_assigned_to_p = number_indices_assigned_to_p * tdma_time_step;
    
    %B^*
    time_to_be_wasted_for_execution = time_assigned_to_p - slot_count*2*C;
    
    %h
    wheel_counter = floor(execution_left / time_to_be_wasted_for_execution);
    
    execution_left = execution_left - wheel_counter*time_to_be_wasted_for_execution;
    
    %find B_j, n such that B <= execution_left
    temp_time = next_slot_starting_time + wheel_counter*time_to_be_wasted_for_execution;
    
    tdmad = [tdma_arbiter tdma_arbiter];
    
    
 b = 0;
 slot_count = 0;
 n_star = 0;


 C_counter = 0;
 slice_position = 0;
 prev_temp = 0;
 temp_exec = 0;
 stallblocks = 0;
 slots_for_p_in_last_cycle = find(tdmad(current_slot+offset:end) == p);
 for y = 1:size(slots_for_p_in_last_cycle,2)
     if(y == 1)
            slot_count = slot_count +1;
     else
         if (slots_for_p_in_last_cycle(y) ~= slots_for_p_in_last_cycle(y-1)+1)
             slot_count = slot_count +1;
             C_counter = C_counter + 2; 
             prev_temp = temp_exec;
             
             stallblocks = stallblocks + (slots_for_p_in_last_cycle(y) - slots_for_p_in_last_cycle(y-1))-1;
         end
         number_slots_so_far = size(slots_for_p_in_last_cycle(1:y),2);
         temp_exec =  number_slots_so_far*tdma_time_step - C_counter*C - C;
         if(temp_exec > execution_left)
            n_star = slot_count;
            b = prev_temp;
            slice_position = slots_for_p_in_last_cycle(y);
            break;
         end
     end
     
 end
 
 %include the time between the next_slot t' and the actual slots considered
 %by B
stallblocks = stallblocks + slots_for_p_in_last_cycle(1)-1;
 
 %b = b - C_counter*C + C;
    
    
%     b = 0;
%     slot_count = 0;
%     n_star = 0;
%     slots_for_p_in_last_cycle = find(tdmad(current_slot+offset:end) == p);
%     for y = 1:size(slots_for_p_in_last_cycle,2)
%        if(y ~= 1)
%             if (slots_for_p_in_last_cycle(y) ~= slots_for_p_in_last_cycle(y-1)+1)
%                 slot_count = slot_count +1;
%                 C_counter = C_counter + 2; 
%             end
%             number_slots_so_far = size(slots_for_p_in_last_cycle(1:y),2);
%             
%             temp_exec =  number_slots_so_far*tdma_time_step - C_counter*C + C;
%             if(temp_exec > execution_left)
%                 b = temp_exec;
%                 n_star = slot_count;
%                 break;
%             end
%        end
%     end
    
    
    consumed_time = wheel_counter*tdma_length+(execution_left - (b-C)) + stallblocks*tdma_time_step + (b-C) +n_star*2*C-C;
    t_advanced = next_starting_slot + consumed_time;
    
    if((wheel_counter*Psi + n_star) >= accesses_left)
        %wcrt = computeWCRTSub(tdma_arbiter, tdma_time_step, p, accesses, execution, time, t_advanced,C);
        wcrt = t_advanced;
        return;
    else
        wcrt = computeWCRTdedicatedPhase(tdma_arbiter, tdma_time_step, t_advanced, p, C, accesses_left - wheel_counter*Psi);
        return;
    end
end
end