function poissonInit

data = 1;
ttInitKernel('prioFP');
ttCreateTask('poissonGenerator', 1, 'poissonGeneratorCode', data);
ttCreateJob('poissonGenerator', ttCurrentTime);

