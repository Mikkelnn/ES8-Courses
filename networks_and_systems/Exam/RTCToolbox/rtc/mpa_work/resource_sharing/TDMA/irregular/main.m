tasks = createProblemInstance('hybrid');
tasks
[tdma number_pe task_to_pe task] = computeTaskPEMapping(tasks);
tdma

p = multipleTasks(tasks);
p
tdma_bus = createTDMABus(1.25, p);


%tdma_bus = old_schedule;
C = 0.05;
tdma_time_step = 2*C;
processor = 2;
%task = 2;
number_of_tasks = size(p(2).tasks, 2);
taskdata = zeros(number_of_tasks, 6);


for j=1:number_of_tasks
task = j;
wcrt = 0;
wcrt_general = 0;
wcrt_dedicated = 0;
wcrt_eles = 0;
accesses = 0;
wcrt_max = 0;
wcrt_general_max = 0;
wcrt_dedicated_max = 0;
execution = 0;


if  (p(processor).tasks(task).period >= size(tdma_bus,2)*tdma_time_step)
    m = p(processor).tasks(task).period / size(tdma_bus,2)*tdma_time_step;
else
    m = size(tdma_bus,2)*tdma_time_step / p(processor).tasks(task).period;
end
    


lcm_counter = 1;

while(lcm_counter <= 7)
    lcm_counter = lcm_counter+1;
    
    wcrt_previous = wcrt;
    wcrt_previous_general = wcrt_general;
    wcrt_pervious_dedicated = wcrt_dedicated;
    wcrt_eles = 0;
    accesses = 0;
    execution = 0;
    
    for i=1:size(p(processor).tasks(task).superblocks,2) 
        i;
        lcm_counter;
        execution_time_upper = p(processor).tasks(task).superblocks(i).execution_time_upper;
        accesses_read = p(processor).tasks(task).superblocks(i).accesses_upper(1);
        accesses_execute = p(processor).tasks(task).superblocks(i).accesses_upper(2);
        accesses_write = p(processor).tasks(task).superblocks(i).accesses_upper(3);

        accesses = accesses + accesses_read  + accesses_execute + accesses_write;
        execution = execution + execution_time_upper;

%        if(lcm_counter ==3 && i == 10)
%            i
%        end
        %hybrid model
        hybrid = 1;
        wcrt1 =  computeWCRTdedicatedPhase(tdma_bus, tdma_time_step, wcrt, processor, C, accesses_read);
        wcrt2 =  computeWCRTexecutionPhase(tdma_bus, tdma_time_step, wcrt1, processor, C, accesses_execute, execution_time_upper);
        wcrt =  computeWCRTdedicatedPhase(tdma_bus, tdma_time_step, wcrt2, processor, C, accesses_write);

        %general model
        general = 1;
        wcrt_general = computeWCRTexecutionPhase(tdma_bus, tdma_time_step, wcrt_general, processor, C, accesses_execute+accesses_read+accesses_write, execution_time_upper);
        
     %   if(wcrt_general < wcrt && i ~= 54)
     %       i
     %       task
     %       lcm_counter
     %       return;
     %   end
%     a = [i lcm_counter wcrt wcrt_general];
%     a;
        
        %dedicated model
        dedicated = 1;
        wcrt1_dedicated = computeWCRTdedicatedPhase(tdma_bus, tdma_time_step, wcrt_dedicated, processor, C, accesses_read);
        wcrt2_dedicated = wcrt1_dedicated + execution_time_upper;
        wcrt_dedicated = computeWCRTdedicatedPhase(tdma_bus, tdma_time_step, wcrt2_dedicated, processor, C, accesses_write+accesses_execute);
        
        wcrt_eles1 = elesanalysis(tdma_bus, 0, accesses_read, wcrt_eles, C, processor);
        wcrt_eles2 = elesanalysis(tdma_bus, execution_time_upper, accesses_execute, wcrt_eles1, C, processor);
        wcrt_eles = elesanalysis(tdma_bus, 0, accesses_write, wcrt_eles2, C, processor);
    end
    
    if((wcrt - wcrt_previous) > wcrt_max)
        wcrt_max = wcrt - wcrt_previous;
    end
    
        
    if((wcrt_general - wcrt_previous_general) > wcrt_general_max)
        wcrt_general_max = wcrt_general - wcrt_previous_general;
    end
    
        
    if((wcrt_dedicated - wcrt_pervious_dedicated) > wcrt_dedicated_max)
        wcrt_dedicated_max = wcrt_dedicated - wcrt_pervious_dedicated;
    end
    
    taskdata(j,:) = [accesses execution+accesses*C wcrt_general_max wcrt_max wcrt_dedicated_max wcrt_eles];
    
    

    
    
    
    
end
end

taskdata
 createfigure(taskdata(2:-1:1,2:end-1)')

