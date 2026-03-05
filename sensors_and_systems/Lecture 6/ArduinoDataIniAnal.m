%ArduinoDataIniAnal makes the simple data analysis for the IMU acceleration and
% rotational speed. Mean, std and the correlation matrix is shown.
% Plots of time and frequency is also made.
% 
% External input: DataFile

% Time-stamp: <2023-03-12 07:33:06 tk>
% Version 1: 2023-02-18 19:37:38 tk Initial version
% Torben Knudsen
% Aalborg University, Dept. of Electronic Systems, Section of Automation
% and Control
% E-mail: tk@es.aau.dk

%% Parameters

fig= 1;
FileName= 'Steady13Hz.dat'; Ts= 1/13;

ma0= [0 0 1];                           % Hypothesis for mean accelerations in g 

%% Definitions etc.

g= 9.8067;                              % Gravitational acceleration
ma0= ma0*g;
d2r= pi/180;                            % Degrees to radians

%% Algorithm

Data= load(FileName);
N= size(Data,1);
T= Data(:,1)/1000;                      % Time in seconds
TDiff= diff(T);                         % Observed sample time in seconds
Ya= Data(:,2:4)*g;                      % Accelerations
Yr= Data(:,5:7)*d2r;                    % Rotational speed

% Plots
figure(fig); fig= fig+1;
histtk(TDiff);
title('Histsogram for sample time');

figure(fig); fig= fig+1;
th= tiledlayout('flow');
nexttile;
plot(T,Ya);
title('Accelerations');
nexttile;
plot(T,Yr);
title('Rotational speed');
title(th,'Time plot');

figure(fig); fig= fig+1;
Legends= {'ax' 'ay' 'az' 'rx' 'ry' 'rz'};
ph= PlotCol([Ya Yr],Legends);
title(ph,'Time plot');

figure(fig); fig= fig+1;
th= tiledlayout('flow');
nexttile;
normplot(Ya);
title('Accelerations');
nexttile;
normplot(Yr);
title('Rotational speed');
title(th,'Normal plots');

figure(fig); fig= fig+1;
th= tiledlayout('flow');
nexttile;
pwelchtk(Ya,[],Ts);
title('Accelerations');
nexttile;
pwelchtk(Yr,[],Ts);
title('Rotational speed');
title(th,'Power spectra');

% Printed results
Lab= ['File: ' FileName ', number of samples : ' int2str(N) ', Date: ' date];
disp(Lab);

Var= Ya;
mv0= ma0;
Res= mean(Var);
Res= [Res; std(Var)];
Res= [Res; mv0];
pvalue= tcdf((Res(1,:)-mv0)./(Res(2,:)/sqrt(N)),N-1);
pvalue= min(pvalue,1-pvalue)*2;
Res= [Res; pvalue];
Res= [Res; min(Var)];
Res= [Res; max(Var)];
Res= [Res; Cov2Corr(cov(Var))];
ColLab= {'y1' 'y2' 'y3'};
RowLab= {'Mean' 'Std' 'Mean-Hyp' 'p-value' 'min' 'max' 'Corr1' 'Corr2' 'Corr3'};
Lab= ['Statistics for accelerations, number of samples : '];
disp(Lab);
disp(array2table(Res,'RowNames',RowLab,'VariableNames',ColLab));

Var= Yr;
mv0= zeros(1,3);
Res= mean(Var);
Res= [Res; std(Var)];
Res= [Res; mv0];
pvalue= tcdf((Res(1,:)-mv0)./(Res(2,:)/sqrt(N)),N-1);
pvalue= min(pvalue,1-pvalue)*2;
Res= [Res; pvalue];
Res= [Res; min(Var)];
Res= [Res; max(Var)];
Res= [Res; Cov2Corr(cov(Var))];
ColLab= {'y1' 'y2' 'y3'};
RowLab= {'Mean' 'Std' 'Mean-Hyp' 'p-value' 'min' 'max' 'Corr1' 'Corr2' 'Corr3'};
Lab= 'Statistics for rotational speed';
disp(Lab);
disp(array2table(Res,'RowNames',RowLab,'VariableNames',ColLab));

% Statistics for the sample time
Res= [mean(TDiff) std(TDiff) min(TDiff) max(TDiff) (T(end)-T(1))/(N-1)];
RowLab= {'Sample time'};
ColLab= {'Mean' 'Std' 'Min' 'Max' '(T(end)-T(1))/(N-1)'};
Lab= 'Statistics for Observed sample time in seconds';
disp(Lab);
disp(array2table(Res,'RowNames',RowLab,'VariableNames',ColLab));
Res= [1/Res(1) (N-1)/(T(end)-T(1))];
RowLab= {'Frequency'};
ColLab= {'1/E(Ts)' '(N-1)/(T(end)-T(1))'};
Lab= 'Statistics for Observed frequency in Hz';
disp(Lab);
disp(array2table(Res,'RowNames',RowLab,'VariableNames',ColLab));
