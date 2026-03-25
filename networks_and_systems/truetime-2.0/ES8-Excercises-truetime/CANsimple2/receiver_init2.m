function receiver_init2

% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority

deadline = 10.0;
ttCreateTask('receiver_task', deadline, 'receiver_code2');

% Network handler 
ttAttachNetworkHandler(1,'receiver_task')

