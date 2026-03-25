function disturbance_init3

data=1;
% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority
ttCreateMailbox('QJamesQ', 69);

ttCreateTask('generator_task', 2, 'generator_code3', data);
ttCreateJob('generator_task', ttCurrentTime);%initial generate msg job

ttCreateTask('sendFromBuffer', 1, 'sendFromBuffer3', data);
ttCreateJob('sendFromBuffer', 0.01); %initial send job



