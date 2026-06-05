

%TDMA PAPER SUPPORT FILES REQUIRED TO GENERATE PROBLEM INSTANCE!!

tasks = createProblemInstance('hybrid');
tasks
p = multipleTasks(tasks);
p
[tdma_bus time_step] = createTDMABus(1.25, p);


considered_task = 1;
period = tasks(considered_task).period;

%sub_tdma = find(tdma_bus == considered_task);

%start_time = (sub_tdma(1) - 1) * C;
%duration = length(sub_tdma) * C;

range = 100;

results = zeros(range, 6);

for m = 1:range
slots_for_unit = 23;
size_schedule = size(p, 2)*3;

%start_position = randi([1, size_schedule - slots_for_unit], 1);
%start_time = (start_position-1) *C;
%duration = slots_for_unit * C;
length = size_schedule * C;

duration = m*(C/2);
%length = duration / 0.8;
start_time = 0;

%SEQENTIAL MODELS
[ wcrt_general_max ...
    wcrt_hybrid_max ...
    wcrt_dedicated_max ...
    wcrt_dedicated_superblocks_max ...
    wcrt_hybrid_superblocks_max ...
    wcrt_general_superblocks_max ...
    wcrt_hybrid_superblocks_max_acq_phases ...
    wcrt_hybrid_superblocks_max_exec_phases ...
    wcrt_hybrid_superblocks_max_rep_phases ] = ...
                                                    SEQ_WCCT(tasks, length, duration, start_time, C, considered_task, period);


%HTS
[wcrt_hybrid_max_seq_triggered wcrt_hybrid_superblocks_max_seq_triggered  wcrt_hybrid_superblocks_max, hybrid_triggered_unsched] = ...
    HTS_WCCT(tasks, length, duration, start_time, C,  wcrt_hybrid_superblocks_max, considered_task, period);


%GTS
[wcrt_general_max_seq_triggered wcrt_general_superblocks_max_seq_triggered wcrt_general_superblocks_max general_triggered_unsched] = ...
    GTS_WCCT(tasks, length, duration, start_time, C, wcrt_general_superblocks_max, considered_task, period);


%HTT
[wcrt_hybrid_superblocks_max_acq_phases wcrt_hybrid_max_trig_triggered wcrt_hybrid_superblocks_max_trig_triggered trig_triggered_unsched] = ...
    HTT_WCCT(tasks, length, duration, start_time, C, wcrt_hybrid_superblocks_max_acq_phases, wcrt_hybrid_superblocks_max_exec_phases, wcrt_hybrid_superblocks_max_rep_phases, considered_task, period);



wcrt_dedicated_max
wcrt_hybrid_max
wcrt_general_max
wcrt_hybrid_max_seq_triggered
wcrt_hybrid_max_trig_triggered
wcrt_general_max_seq_triggered


results(m,:) = [wcrt_dedicated_max
wcrt_hybrid_max
wcrt_general_max
wcrt_hybrid_max_seq_triggered
wcrt_hybrid_max_trig_triggered
wcrt_general_max_seq_triggered];

% 
%  example20msB.blocks = tasks(considered_task).superblocks;
%  example20msB.period = period;
%  example20msB.slots_for_unit = slots_for_unit;
%  example20msB.size_schedule = size_schedule;
%  example20msB.start_time = start_time;
%  example20msB.results =  results;
%  save example20Bosch.mat example20msB
end
