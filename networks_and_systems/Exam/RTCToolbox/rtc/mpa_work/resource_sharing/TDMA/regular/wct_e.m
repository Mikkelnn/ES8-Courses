function t = wct_e( length, duration, start_time, accesses, current_time, C, computations)

if(duration < C)
    t = Inf;
    return;
end

if accesses == 0
    t = current_time + computations;
    return;
end

t_relative = ((current_time) - ((floor(current_time / length)) * length));

current_slot_active = 0;
if t_relative < start_time + duration
    if (t_relative >= start_time) %numerical problem !!
        current_slot_active = 1;
    end
end

if current_slot_active == 0
    
    if t_relative < start_time
        t_next = start_time - t_relative;
    else
        t_next = length - (t_relative - start_time);
    end
%     if(t_next < 10^-6)
%         t_next = 0;
%     end
    
    t1 = 0;
    if computations > t_next
        t1  = wct_e(length, duration, start_time, accesses, current_time+t_next, C, computations-t_next);
    end
    t2 = wct_e(length, duration, start_time, accesses-1, current_time+t_next, C, computations);
    t = max(t1,t2);
    return;
else
    t_offset = t_relative - start_time;
    
    if( (duration - t_offset - C) > computations )
        t = wct_ar(length, duration, start_time, accesses, current_time+computations+C, C);
        return
    else
        computations_remaining = computations - (duration - t_offset - C);
        %accesses = accesses - 1;
        t = current_time + length - t_offset;
        
        blocks = floor(computations_remaining / max((duration - 2*C), 0) );
        
        if blocks > accesses 
            t = t + accesses*length+C + ( computations_remaining - accesses * max((duration - 2*C), 0) );
            return;
        end
        
        t = t + blocks * length + ( computations_remaining - blocks * max((duration - 2*C), 0) );
        %t = t + length + C;
        accesses = accesses - blocks;
        t = wct_ar(length, duration, start_time, accesses, t, C);
        return;
    end
            


end

end

