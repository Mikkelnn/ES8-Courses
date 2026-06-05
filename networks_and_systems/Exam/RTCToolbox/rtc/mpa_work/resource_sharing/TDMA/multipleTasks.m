function [p] = multipleTasks(tasks)
number_of_tasks = length(tasks);

number_of_processing_elements = randi([ 2, number_of_tasks-1], 1);


%number_of_tasks_per_pe = ceil(number_of_tasks / number_of_processing_elements)
number_of_tasks_per_pe = 1;




i =1;
tasks_on_pe_counter = 1;
for j = 1:number_of_tasks
    p(i).tasks(tasks_on_pe_counter) = tasks(j);
    p(i).W = tasks(j).period;
    %if(tasks_on_pe_counter == 1)
    %    p(i).W = tasks(tasks_on_pe_counter).period;
    %else
    %    p(i).W = [p(i).W p(i).W(tasks_on_pe_counter-1)+p(i).tasks(tasks_on_pe_counter-1).period];
    
    %end
    tasks_on_pe_counter = tasks_on_pe_counter + 1;
    if(tasks_on_pe_counter-1 == number_of_tasks_per_pe )
        tasks_on_pe_counter = 1;
        i = i+1;
    end
    
end

end