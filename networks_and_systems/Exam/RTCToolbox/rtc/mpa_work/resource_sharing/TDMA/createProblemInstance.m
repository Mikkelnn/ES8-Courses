function [tasks] = createProblemInstance(use_case)



%number_of_tasks = randi(5, 1, number_of_processing_elements);

number_of_tasks = 9;
tasks_periods = [1.25 2.5 5 10 20 40 120 240 1200];
lower_number_superblocks = [1 2 8 75 110 18 30 50 15];
upper_number_superblocks = [3 4 9 90 130 25 40 65 20];
upper_access_read = [0 1 0.3 0.66 1.8 0.5 1.6 0.05 0];
lower_access_read = [0 0.33 0.2 0.5 1.4 0.25 1.4 0 0];
upper_access_write = [0 0 0.3 0.33 1.8 0.5 1 0 0.5];
lower_access_write = [0 0 0.2 0.25 1.5 0.25 0.7 0 0.3];
upper_exec = [0.0275 0.016 0.045 0.035 0.05 0.06 0.27 0.2 0.5];
lower_exec = [0.025 0.012 0.03 0.03 0.04 0.05 0.23 0.16 0.4];




lower_number_superblocks(5) = 50;
upper_number_superblocks(5) = 50;
upper_access_read(5) = 1.5;
lower_access_read(5) = 1.4;
upper_access_write(5) = 1.52;
lower_access_write(5) = 1.48;
upper_exec(5) = 0.055;
lower_exec(5) = 0.042;

tasks = struct('superblocks', 'number_of_superblocks');


%create the number for an AutoSAR application
for j=1:number_of_tasks
    number_of_superblocks = randi([lower_number_superblocks(j) upper_number_superblocks(j)],1);
    superblocks = struct;
    
    
    
    upper = upper_access_read(j);
    lower = lower_access_read(j);
    upper_read = ceil((lower + (upper-lower).*rand(1)));
    lower_read = ceil((lower + (upper-lower).*rand(1)));
    upper = upper_access_write(j);
    lower = lower_access_write(j);
    upper_write = ceil((lower + (upper-lower).*rand(1)));
    lower_write = ceil((lower + (upper-lower).*rand(1)));
    
    
    distribution_read = randperm( number_of_superblocks);
    distribution_write = randperm( number_of_superblocks);
    distribution_execute = randperm( number_of_superblocks);
    
    if(use_case == 'hybrid')
        exec_accesses_per_superblock = ceil(0.2*upper_read + 0.2*upper_write);
        upper_read = upper_read - upper_read*0.2;
        upper_write = upper_write - upper_write*0.2;
        read_accesses_per_superblock = ceil( upper_read);
        write_accesses_per_superblock = ceil( upper_write);    
    else
        read_accesses_per_superblock = ceil( upper_read);
        write_accesses_per_superblock = ceil( upper_write);
        exec_accesses_per_superblock = 0;
    end
    
    upper_read_task = upper_read * number_of_superblocks;
    upper_write_task = upper_write * number_of_superblocks;
    upper_exec_task = upper_exec * number_of_superblocks;
    
    %or upper is lower then the number of superblocks => which blocks not
    %to assign accesses
    
    
    for x=1:number_of_superblocks
       lower = lower_exec(j);
       upper = upper_exec(j);
       
       k = distribution_execute(x);
       
       
       superblocks(k).execution_time_lower = lower + (upper-lower).*rand(1);
       lower = superblocks(k).execution_time_lower;
       superblocks(k).execution_time_upper = lower + (upper-lower).*rand(1);
       

       superblocks(k).accesses_lower = zeros(1,3);
       superblocks(k).accesses_upper = zeros(1,3);
       
       if(upper_exec_task > 0)
           superblocks(k).accesses_upper(3) = exec_accesses_per_superblock;
           upper_exec_task = upper_exec_task - exec_accesses_per_superblock;
       end
       
       k = distribution_read(x);
       if(upper_read_task > 0)
           superblocks(k).accesses_upper(1) = read_accesses_per_superblock;
           upper_read_task = upper_read_task - read_accesses_per_superblock;
       end
       
       k = distribution_write(x);
       if(upper_write_task > 0)
           superblocks(k).accesses_upper(2) = write_accesses_per_superblock;
           upper_write_task = upper_write_task - write_accesses_per_superblock;
       end

       
    end

    tasks(j).superblocks = superblocks;
    tasks(j).number_of_superblocks = number_of_superblocks;
    tasks(j).period = tasks_periods(j);

end
  

end

