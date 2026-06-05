
function tdma = createTDMAArbiter(processing_elements)
%processing_elements is an array of arbitary size,
%each element representing a single processing element.
%e.g.
% PE1 = processing_elements(1);
%   PE1.C = atomic time unit (i.e. time required to process one single
%   memory access)
%   PE1.period = period of the static time wheel that executes on this
%   element
%   PE1.tasks = an array containing the tasks that execute on a processing
%   element.
%       task1 = PE1.tasks(1);
%       task1.activation = relative activation time
%       task1.superblocks = an array containing the superblocks a task is
%       constituted of
%       superblock1 = task1.superblocks(1);
%           superblock1.type = the superblocks type (dedicated, general,
%           hybrid)
%       superblock1.phases = the superblocks phases (3 in dedicated and
%       hybrid, 1 in general)
%       superblock1.execution_time = tuple with lower and upper bound on
%       execution time
%       superlbock1.accesses = one tuple for each phase, representing the
%       lower and upper bound on accesses to the shared memory, e.g.,
%       (1) dedicated phases: 
%           superblock1.accesses = [r_l, r_u, 0, 0, w_l, w_u]
%       (2) general memory
%           superblock1.accesses = [\mu_l, \mu_u]
%       (3) hybrid model
%           superblock1.accesses =  [r_l, r_u, \mu_l, \mu_u, w_l, w_u]
%
%The output is an array with Hyperperiod/Ci elements. Therefore, each
%element represents an atomic time step and holds the ID of the processing
%element it is assigned to. E.g. tdma[4] = 5, represents the point in time
%equal to 4Ci and processing element 5 is granted access to the bus from
%4Ci to 5Ci.


number_of_processing_elements = length(processing_elements);
hyperperiod = processing_elements(1).period; 
for i=1:number_of_processing_elements
    hyperperiod = lcm(hyperperiod, processing_elements(i).period);
end
tdma = zeros(1, hyperperiod);

accesses = 0;
processing_element_accesses = zeros(1, number_of_processing_elements);

for i = 1:number_of_processing_elements
    for j= 1:length(processing_elements(i).tasks
        for k = 1:length(processing_elements(i).tasks(j).superblocks)
            accesses = accesses + processing_elements(i).tasks(j).superblocks(k);
            processing_element_accesses[i] = processing_element_accesses[i] + processing_elements(i).tasks(j).superblocks(k);
        end
    end
end

bandwidth_share = processing_element_accesses./accesses







end