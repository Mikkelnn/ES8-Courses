function [found starting_time] = getNextTimeSlotForP(tdma,p,current_slot, time_step)
%returns the number of elements in an array tdma, one has to go forward to
%find the next time slot allocated to p. If the current time slot is
%already assigned to p, the output is 0; This can cross wheel boundaries.



tdmad = [tdma tdma];

found = 0;
if(tdmad(current_slot) == p)
    in_slot = 1;
end

for i = current_slot:size(tdmad,2)
    if(tdmad(i) == p)
        if(~in_slot)
            found = i;
            break;
        end
    else
        in_slot = 0;
    end
    
end

if(found > size(tdma,2))
    found = found - size(tdma,2);
    found = found - current_slot;
    starting_time = (found) * time_step + size(tdma,2)*time_step;
    
else
    found = found - current_slot;
    starting_time = (found) * time_step;
end




    
end