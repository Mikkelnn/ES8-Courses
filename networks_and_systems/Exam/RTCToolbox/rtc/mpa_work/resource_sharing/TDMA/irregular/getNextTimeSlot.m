function [index next_slot_starting_time] = getNextTimeSlot(tdma,current_slot, time_step)
%Returns the offset from the current_slot to the next time slot which is
%assigned to any other pe than the current slot. The value offset can cross
%boundaries. 

p = tdma(current_slot);
tdmad = [tdma tdma];

indices = find(tdmad(current_slot+1:end) ~= p);
offset = indices(1);

if(isempty(offset))
    error('There is no time slot for p=%d in the tdma wheel', p);
else
    index = offset;
end


if( (current_slot + offset) > size(tdma,2))
    next_slot = offset - size(tdma,2);
    next_slot_starting_time = next_slot * time_step;
    next_slot_starting_time = next_slot_starting_time +  size(tdma,2)*time_step;
else
    next_slot =offset;
    next_slot_starting_time = next_slot * time_step;
end

    
end