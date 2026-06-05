function [tdma processing_elements task_to_pe task] = computeTaskPEMapping(tasks)

%map the application to the processing elements
number_of_tasks = size(tasks, 2);
processing_element_counter = 1;
atomic_unit = 1.25;
max_period = tasks(number_of_tasks).period;

C = 0.04;
task_to_pe =  zeros(1, number_of_tasks);

%every item of this array represents one slot of 1.25ms
%in the loop we sum up the elements that are assigned that slot.
%if an element increase 1.25, it is infeasible
%if an element is more than 50% we do not proceed to map superblock there

last_task = 1;
all_done = 0;
infeasible = 0;
i = 1;

skipped_due_to_task = zeros(1, number_of_tasks);



while(~all_done)

    pe_tdma = zeros(1, max_period / atomic_unit);
    start_new_pe = 0;


    for j=last_task:number_of_tasks
        
        pe_tdma_task = zeros(1, max_period / atomic_unit);
        tasks_pe_mapping_temp = zeros(1, length(pe_tdma));
        tasks_pe_mapping = zeros(1, length(pe_tdma));
        
        superblocks_pe_mapping  = zeros(tasks(j).number_of_superblocks, length(pe_tdma));
        superblocks_pe_mapping_temp  = zeros(tasks(j).number_of_superblocks, length(pe_tdma));

        %sum all the accesses that we have to host
        current_tasks_access_time = 0;
        for k=1:tasks(j).number_of_superblocks
            current_tasks_access_time = current_tasks_access_time + sum(tasks(j).superblocks(k).accesses_upper);
            current_tasks_access_time = current_tasks_access_time + tasks(j).superblocks(k).execution_time_upper;
        end
        current_tasks_access_time = current_tasks_access_time * C;


        
        %how much atomic units can we use to pack the superblocks
        superblocks_per_unit_rel = (atomic_unit) / tasks(j).period;
        
        %the amount of time for accesses that have to fit in one atomic unit
        %if the period of the task is 1.25, the superblocks_per_unit_rel is 1
        %and therefore all the superblocks accesses have to fit here. As a
        %result superblock_time_per_unit equals the sum of accesses multiplied
        %with C for this particular superblock.
        superblock_time_per_unit = current_tasks_access_time * superblocks_per_unit_rel;


        %task assigning superblocks to time slots
        current_tasks_access_time_op = current_tasks_access_time;
        window = tasks(j).period / atomic_unit;
        
        

        for m=1:window:length(pe_tdma)
            k = 1;
            
            for p=m:m+window-1
                
                %assign superblocks in the amount of
                %superblock_time_per_unit
                accumulated_assigned_time = 0;
                while(accumulated_assigned_time <= superblock_time_per_unit)
                    if(k > tasks(j).number_of_superblocks)
                        break;
                    end
                   %find the sequence of superblocks that stays below
                   %superblock_time_per_unit
                   %if this number=0, test if the first superblock would
                   %fit, if that is the case = assign it, else go on
                   %without assigning a superblock. This makes sure that
                   %bigger junks of accesses are also assigned. Assigning
                   %them in the earliest stage increases feasibility, since
                   %with increasing task index, also the period increases
                   %and therefore there are more chance to assign the
                   %succeeding superblocks.
                   superblock_upper_total_time = sum(tasks(j).superblocks(k).accesses_upper)*C + tasks(j).superblocks(k).execution_time_upper;
                   
                   limit = atomic_unit * 0.75;
                   %if there was already a new processing element assigned, and the limit still does not fit,
                   %allow up to the complete unit;
                   if( pe_tdma(p) + superblock_upper_total_time > limit && skipped_due_to_task(j) )%skipped due to this task ?) 
                      limit = atomic_unit;
                   end
                   
                   if(pe_tdma(p) + superblock_upper_total_time < limit)
                       pe_tdma(p) = pe_tdma(p) + superblock_upper_total_time;
                       pe_tdma_task(p) = pe_tdma_task(p) + superblock_upper_total_time;
                       current_tasks_access_time_op = current_tasks_access_time_op - superblock_upper_total_time;
                       accumulated_assigned_time = accumulated_assigned_time + superblock_upper_total_time;
                       tasks_pe_mapping(p) = 1;
                       tasks_pe_mapping_temp(p) = 1;
                       superblocks_pe_mapping(k, p) = superblock_upper_total_time;
                       superblocks_pe_mapping_temp(k, p) = superblock_upper_total_time;
                       %superblocks_pe_assignment(k, p) = 1;
                       %superblocks_pe_assignment_temp(k, p) = 1;
                       k = k+1;
                   else
                      break; 

                   end

                end
            end
            if(current_tasks_access_time_op > 0.000001) %numerical problems !!
               %this tasks cannot fit on this pe
               %1 stop assigning tasks here
               %2 open a new pw
               processing_element_counter = processing_element_counter + 1;
               %3 restart assigning tasks on the new pe, from this task
               pe_tdma = pe_tdma - pe_tdma_task;
               tasks_pe_mapping = tasks_pe_mapping - tasks_pe_mapping_temp;
               superblocks_pe_mapping = superblocks_pe_mapping - superblocks_pe_mapping_temp;
               %superblocks_pe_assignment =  superblocks_pe_assignment -  superblocks_pe_assignment_temp;
               if(sum(pe_tdma) == 0)
                   infeasible = 1
               end
               start_new_pe = 1
               skipped_due_to_task(j) = 1;
               last_task = j;
            end
            
            %m = p; %do not overlap
            current_tasks_access_time_op = current_tasks_access_time;
            if(start_new_pe)
                break;
            end
            if(infeasible)
                break;
            end
        end
        if(start_new_pe)
            %j
            break;
        else
            task_to_pe(j) = processing_element_counter;
        end
        if(infeasible)
            break;
        end
        task(j).tasks_pe_mapping = tasks_pe_mapping;
        task(j).superblocks_pe_mapping = superblocks_pe_mapping;
        %task(j).superblocks_pe_assignment = superblocks_pe_assignment;
        

    end
    tdma(i).schedule = pe_tdma;
    %tdma(i).tasks_superblocks_mapping = tasks_superblocks_mapping;
    if(~start_new_pe)
        all_done = 1;
    end
    if(infeasible)
        break;
    end
    i = i+1;
end
    
processing_elements = processing_element_counter;

end
