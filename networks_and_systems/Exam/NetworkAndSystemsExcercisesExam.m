%% NetworkAndSystemsExcercisesExam
% MATLAB conversion of the Maple worksheet NetworkAndSystemsExcercisesExam.mw.
% The original worksheet defines the constants below, evaluates three
% calculations, and creates the same 18 plots represented in the worksheet.

clear;
clc;
close all;

%% Settings
% Set savePlots to true to export every generated figure as a PNG file.
savePlots = false;
outputDir = 'NetworkAndSystemsExcercisesExam_plots';

if savePlots && ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Constants copied from the Maple worksheet
BW = 1e6;
nSensors = 8;
fs = 1000;
Overhead = 100;
sampleBytes = 100;

%% Equivalent calculations for the Maple solve(...) calls
% Maple: solve(BW = (x + Overhead) * nSensors * fs, x)
payloadPerSensorSample = BW / (nSensors * fs) - Overhead;

% Maple definitions:
%   Data(t) := 100*t
%   SentData(t) := 125*t - 100
Data = @(t) sampleBytes .* t;
SentData = @(t) 125 .* t - 100;

% Maple: solve(Data(t) = SentData(t), t)
crossoverTime = Overhead / (125 - sampleBytes);

% Maple: solve(BW = ((x*100 + Overhead) * nSensors * fs) / x, x)
denominator = BW - sampleBytes * nSensors * fs;
if denominator == 0
    error('No finite packet size satisfies the bandwidth equation.');
end
samplesPerPacket = Overhead * nSensors * fs / denominator;

fprintf('Payload per sensor sample: %g\n', payloadPerSensorSample);
fprintf('Data(t) and SentData(t) cross at t = %g\n', crossoverTime);
fprintf('Samples per packet satisfying BW: %g\n', samplesPerPacket);

%% Functions used by the worksheet plots
aggregateBandwidth = @(x) ((x .* sampleBytes + Overhead) .* nSensors .* fs) ./ x;
packetizedData = @(t, n) floor(t ./ n) .* (n .* sampleBytes + Overhead) .* nSensors;
maxRate = @(t) (BW / 1000) .* t;

%% Maple: plot([Data(t), SentData(t)], t = 0 .. 10)
t = linspace(0, 10, 501);
figure;
plot(t, Data(t), 'DisplayName', 'Data(t) = 100t');
hold on;
plot(t, SentData(t), 'DisplayName', 'SentData(t) = 125t - 100');
hold off;
xlabel('t');
ylabel('Data');
title('Data and sent data');
grid on;
legend('show');
if savePlots
    saveas(gcf, fullfile(outputDir, '01_data_vs_sent_data.png'));
end

%% Maple: plot([1e6, ((x*100 + Overhead)*8*1000)/x], x = 1 .. 10)
x = linspace(1, 10, 501);
figure;
plot(x, BW .* ones(size(x)), 'DisplayName', 'BW');
hold on;
plot(x, aggregateBandwidth(x), 'DisplayName', 'Aggregate bandwidth');
hold off;
xlabel('Samples per packet (x)');
ylabel('Bandwidth');
title('Bandwidth requirement versus packet size');
grid on;
legend('show');
if savePlots
    saveas(gcf, fullfile(outputDir, '02_bandwidth_vs_packet_size.png'));
end

%% Maple loop: plot([maxRate(t), F[n]], t = 0 .. 50), for n = 1 .. 8
t = linspace(0, 50, 2501);
for n = 1:nSensors
    figure;
    plot(t, maxRate(t), 'DisplayName', 'maxRate(t)');
    hold on;
    plot(t, packetizedData(t, n), 'DisplayName', sprintf('F[%d](t)', n));
    hold off;
    xlabel('t');
    ylabel('Data');
    title(sprintf('Aggregate packetized data, n = %d', n));
    grid on;
    legend('show');
    if savePlots
        saveas(gcf, fullfile(outputDir, sprintf('03_aggregate_packetized_n_%d.png', n)));
    end
end

%% Maple loop: plot([maxRate(t), F[n]/8], t = 0 .. 50), for n = 1 .. 8
for n = 1:nSensors
    figure;
    plot(t, maxRate(t), 'DisplayName', 'maxRate(t)');
    hold on;
    plot(t, packetizedData(t, n) ./ nSensors, ...
        'DisplayName', sprintf('F[%d](t) / %d', n, nSensors));
    hold off;
    xlabel('t');
    ylabel('Data');
    title(sprintf('Per-sensor packetized data, n = %d', n));
    grid on;
    legend('show');
    if savePlots
        saveas(gcf, fullfile(outputDir, sprintf('04_per_sensor_packetized_n_%d.png', n)));
    end
end
