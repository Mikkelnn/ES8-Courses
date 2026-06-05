load('C:\Users\andrschr.NB-10112\Documents\svn\paper\dac2010\implementation\example5.mat');
number_of_blocks_5 = size(example5ms.blocks,2);

S = zeros(number_of_blocks_5, 2);


for j=1:number_of_blocks_5
    S(j, 1) = round(example5ms.blocks(j).execution_time_upper*100);
    S(j, 2) = sum(example5ms.blocks(j).accesses_upper);
end

