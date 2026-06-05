function wcrt = computeWCRTdedicatedPhase(tdma_arbiter, tdma_time_step, time, p, C, accesses)

tdma_length = size(tdma_arbiter,2) * tdma_time_step;

cycles_length = floor(time / tdma_length);
rel_time = time - cycles_length*tdma_length;

wheel_correction = 0;

current_slot = ceil(rel_time / tdma_time_step);
if(current_slot == 0)
    current_slot = 1;
end


next_slot = find(tdma_arbiter(current_slot:end) ~= tdma_arbiter(current_slot));
if(isempty(next_slot))
    next_slot = find (tdma_arbiter(1:end) ~= p);
    wheel_correction = 1;
end
    
next_slot = next_slot(1) + current_slot-1 -  wheel_correction*size(tdma_arbiter,2);
if(next_slot <= 0)
    next_slot = 1;
end


lambda = size(find(tdma_arbiter == p),2); %performed access per cycles


next_slot_start = (next_slot-1) * tdma_time_step;

if(tdma_arbiter(current_slot) == p)
    theta = 1;
else
    theta = 0;
end


if ((floor( ((next_slot_start) - rel_time) / C) * theta) >= accesses)
    rel_time = rel_time+accesses*C;
else
    
    left_accesses = accesses - (floor( (next_slot_start - rel_time) / C) * theta);
    required_cycles = floor(left_accesses / lambda); 

    next_slot_start = next_slot_start + required_cycles*tdma_length;
    left_accesses = left_accesses - (required_cycles*lambda);
    
    temp_tdma = [tdma_arbiter tdma_arbiter];
    %next_slot
    
    count_atomic_slots_array = find (temp_tdma(next_slot:end) == p);
    index_in_overall_tdma = count_atomic_slots_array(left_accesses);
    

    rel_time = next_slot_start + index_in_overall_tdma*C;
    

    %the relative time in a time cycle stays the same, since we only add
    %complete cycles 
end

wcrt = rel_time + cycles_length*tdma_length;


end