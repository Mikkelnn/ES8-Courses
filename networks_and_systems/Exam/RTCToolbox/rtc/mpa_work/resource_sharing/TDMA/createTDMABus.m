function [tdma_arbiter time_step]= createTDMABus(atomic, p)
%create a tdma bus, granting each pe as much share on the bandwidth as its
%acceesses have on the total number of accesses to the shared resource
%time slots for pe as continues, e.g., if a slot has 20% of the slots, the
%number of slots that total for 20% are in one sequence. As a results each
%pe has one extended slot per TDMA cycle and the slots for a processor are
%equally spaced. 



time_step = atomic;

number_pe = size(p,2);
peShare = zeros(1,number_pe);

for i =1:number_pe
    number_of_tasks = size(p(i).tasks,2);
    tasks_accesses = 0;
    for j = 1:number_of_tasks
        number_of_superblocks = size(p(i).tasks(j).superblocks,2);
        for k =1:number_of_superblocks
             accesses = sum(p(i).tasks(j).superblocks(k).accesses_upper);
             tasks_accesses =  tasks_accesses + accesses;
        end
    end
    peShare(i) = tasks_accesses;
    
end

peShare = peShare ./ sum(peShare);




slot_frequency_per_pe = 1./peShare;
slot_frequency_per_pe(find(slot_frequency_per_pe==Inf)) = 0;
window = sum(slot_frequency_per_pe);
slot_assignment = ceil(window./slot_frequency_per_pe);
slot_assignment(find(slot_assignment==Inf)) = 0;

length_w = sum(slot_assignment);
tdma_arbiter = zeros(1, length_w);

for i = 1:ceil(window):length_w-ceil(window)+1
    position = 0;
    for j = 1:number_pe
        tdma_arbiter(i+position:i+slot_assignment(j)+position-1) = j;
        position = position + slot_assignment(j);
    end
end






end