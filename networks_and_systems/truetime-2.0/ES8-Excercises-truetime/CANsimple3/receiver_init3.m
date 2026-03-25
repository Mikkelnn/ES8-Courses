function receiver_init3

% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority

deadline = 10.0;
ttCreateTask('receiver_task', deadline, 'receiver_code3');

% Network handler 
ttAttachNetworkHandler(1,'receiver_task')

