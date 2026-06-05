function wcrt = computeWCRTSub(tdma_arbiter, tdma_time_step, p, accesses, execution, time, upper_bound, C)


tdma_length = size(tdma_arbiter,2) * tdma_time_step;
cycles_length = floor(time / tdma_length);
rel_time = time - cycles_length*tdma_length;


current_slot = ceil(rel_time / tdma_time_step);
starting_time_current_time_slot = (current_slot-1) * tdma_time_step + cycles_length*tdma_length;


[offset next_slot_starting_time]= getNextTimeSlot(tdma_arbiter, current_slot, tdma_time_step);
next_starting_slot = starting_time_current_time_slot+next_slot_starting_time;

 %tdmad = [tdma_arbiter tdma_arbiter];
 indices_slots_assigned_to_p = find(tdma_arbiter == p);
 number_indices_assigned_to_p = size(indices_slots_assigned_to_p,2);
 %find the number of incontinuous section = this is the number of slots
 %then;
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
 
 wheels_required = floor(accesses / Psi);
  accesses_remaining = accesses - wheels_required*slot_count;
 
 tdmad = [tdma_arbiter tdma_arbiter];
    
 b = 0;
 slot_count = 0;
 n_star = 0;
 temp_exec = 0;
 previous_exec =0;
 C_counter = 0;
 slice_position = 0;
 slots_for_p_in_last_cycle = find(tdmad(current_slot+offset:end) == p);
 for y = 1:size(slots_for_p_in_last_cycle,2)
     if(y == 1)
            slot_count = slot_count +1;
     else
         if (slots_for_p_in_last_cycle(y) ~= slots_for_p_in_last_cycle(y-1)+1)
             slot_count = slot_count +1;
             C_counter = C_counter + 2; 
             previous_exec = temp_exec;
         end
         number_slots_so_far = size(slots_for_p_in_last_cycle(1:y),2);
         temp_exec =  number_slots_so_far*tdma_time_step;
         if(slot_count > accesses_remaining)
            n_star = slot_count;
            b = previous_exec;
            slice_position = slots_for_p_in_last_cycle(y);
            break;
         end
     end
     
 end
 
 b = b - C_counter*C + C;
 
 
 %b!!
 
 
 %B*
 time_assigned_to_p = number_indices_assigned_to_p * tdma_time_step;
 time_to_be_wasted_for_execution = time_assigned_to_p - Psi*2*C;
 
 execution_temp = execution - time_to_be_wasted_for_execution*floor(accesses/Psi) - b - (next_starting_slot - time - C);
 
 %slice_position = slice_position+current_slot+offset;
 
 used_wheels = floor(slice_position / size(tdma_arbiter,2));
 slice_position = slice_position - used_wheels*size(tdma_arbiter,2);
 
 
 
 [index time_extension] = getNextTimeSlot(tdma_arbiter,current_slot, tdma_time_step);
 if(n_star > current_slot+index)
     
     time_temp = next_starting_slot + wheels_required * tdma_length + (slice_position+((current_slot+offset)-1))*tdma_time_step - ((current_slot + next_slot_starting_time)-1)*tdma_time_step;
 else
     time_temp = next_starting_slot + wheels_required * tdma_length + (slice_position+((current_slot+offset)-1))*tdma_time_step + tdma_length - (current_slot + next_slot_starting_time)*tdma_time_step;
 end
 
 lower_bound = rel_time + time_temp + execution_temp;

 if(lower_bound > upper_bound)
     temp = lower_bound;
     lower_bound =  upper_bound;
     upper_bound = temp;
 end

 

 step_size = C;
 done = 0;
 %do the binary search
 step_size = C;
 last_feasible = 0;

 bound = upper_bound;
 while(~done)
     
     [feasibility offset] = worstCaseTest(tdma_arbiter, tdma_time_step, p, execution, accesses, time, bound, C);
     
     if(~feasibility)
         lower_bound = bound;
         bound = bound + (upper_bound - bound )/2;
         if (bound == lower_bound)
             bound = bound + lower_bound;
         end
         
     else
         upper_bound = bound; 
         last_feasible = bound;
         bound = bound - (bound - lower_bound)/2;
     end
     
     if(abs(last_feasible - bound) <= step_size)
         bound = last_feasible;
         done=1;
     end
 
 end
 
 wcrt = bound;
 
  
 


end