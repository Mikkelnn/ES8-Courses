function [p ] = singleTask(tasks)
   number_of_tasks = length(tasks);
   
   number_of_processing_elements = number_of_tasks;
   
   
   for j=1:number_of_tasks
       p(j).tasks(1).superblocks = tasks(j).superblocks;
       p(j).tasks(1).W = [0];
   end
end