function poissonReceiveInit

% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority

deadline = 10.0;
ttCreateTask('receiver_task', deadline, 'poissonReceiveCode');

% Network handler 
ttAttachNetworkHandler(1,'receiver_task')
