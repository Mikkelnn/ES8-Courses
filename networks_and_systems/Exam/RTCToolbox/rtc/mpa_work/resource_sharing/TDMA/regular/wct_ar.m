function t = wct_ar(length, duration, start_time, accesses, current_time, C)

if(duration < C)
    t = Inf;
    return;
end

if(accesses == 0)
    t= current_time;
    return;
end

t_relative = current_time - floor(current_time / length) * length;

n = floor(duration / C);

if t_relative < start_time + duration
    if t_relative > start_time
        access_current_cycle = floor( (start_time + duration - t_relative) / C );
        if access_current_cycle >= accesses
            t = current_time+accesses*C;
            return;
        else
            accesses = accesses - access_current_cycle;
        end
    end
    
end

if t_relative <= start_time
    t_next = start_time - t_relative;
else
    t_next = length - (t_relative - start_time);
end


cycles = max(ceil(accesses / n) - 1, 0);
accesses = accesses -  cycles*n;

if(accesses > n)
    t = current_time + t_next + length * (cycles + 1) + (accesses - n) * C;
    return;
else
    t = current_time + t_next + length * cycles + accesses * C;
    return;
end


end