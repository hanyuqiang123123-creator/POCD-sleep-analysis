%% Extract source data for Fig. 5H (representative sigma dynamics and spindles)
% This script performs data extraction only. Final plotting is done in Python.

clear; clc;

edfPath = 'F:\1.Sleep\eXdata\EEG\EEG_CON\NO.06_1212_14594\6P\6P1\exportx_200Hz.edf';
scorePath = 'F:\1.Sleep\eXdata\EEG\EEG_CON\NO.06_1212_14594\6P\6P1\export_scores_n.tsv';
spindlePath = 'F:\1.Sleep\EEGsoftware\EEGSoftware\SpindleSO\DATA\Detections\0414singal\CONT\Spi_NO06121214594_6P_6P1_0min-1440min_200HzCh1.mat';
outputDir = 'F:\1.Sleep\PHD稿件\Figures\Fig5\data';

addpath('F:\1.Sleep\EEGsoftware\EEGSoftware\AutoFQ');
addpath('F:\1.Sleep\EEGsoftware\EEGSoftware\Spindle');

windowStartMin = 0;
windowEndMin = 30;
nremCode = 2;

[hdr, record] = edfread(edfPath);
if isfield(hdr, 'frequency') && ~isempty(hdr.frequency)
    fs = double(hdr.frequency(1));
elseif isfield(hdr, 'samples') && isfield(hdr, 'duration')
    fs = double(hdr.samples(1)) / double(hdr.duration);
else
    error('Unable to determine EDF sampling rate.');
end

eeg = double(record(1, :));

idxStart = round(windowStartMin * 60 * fs) + 1;
idxEnd = min(round(windowEndMin * 60 * fs), numel(eeg));
y = eeg(idxStart:idxEnd);

scoreTable = readtable(scorePath, 'Delimiter', '\t', 'FileType', 'text');
scoreValues = scoreTable{6:end, 5};
scoreStart = windowStartMin * 12 + 1;
scoreEnd = min(windowEndMin * 12, numel(scoreValues));
windowScores = scoreValues(scoreStart:scoreEnd);
timeSec = (0:numel(y)-1)' / fs;

trialinfo = double(h5read(spindlePath, '/EEG_Ch1/trialinfo'));
if size(trialinfo, 2) == 13
    spindleStartAbs = trialinfo(:, 2);
    spindleEndAbs = trialinfo(:, 3);
elseif size(trialinfo, 1) == 13
    spindleStartAbs = trialinfo(2, :)';
    spindleEndAbs = trialinfo(3, :)';
else
    error('Unexpected trialinfo dimensions.');
end

eventMask = spindleEndAbs >= idxStart & spindleStartAbs <= idxEnd;
eventStartMin = max((spindleStartAbs(eventMask) - idxStart) / fs / 60, 0);
eventEndMin = min((spindleEndAbs(eventMask) - idxStart) / fs / 60, windowEndMin-windowStartMin);
eventCenterMin = (eventStartMin + eventEndMin) / 2;

traceTable = table(timeSec, y(:), repmat(fs, numel(y), 1), ...
    'VariableNames', {'time_sec','eeg_uv','sampling_rate_hz'});
eventTable = table(eventStartMin, eventEndMin, eventCenterMin, ...
    'VariableNames', {'start_min','end_min','center_min'});
scoreEpoch = (1:numel(windowScores))';
scoreStartSec = (scoreEpoch - 1) * 5;
scoreEndSec = scoreEpoch * 5;
scoreOut = table(scoreEpoch, scoreStartSec, scoreEndSec, windowScores(:), ...
    'VariableNames', {'epoch_1based','start_sec','end_sec','stage_code'});

writetable(traceTable, fullfile(outputDir, 'Fig5H_raw_EEG_6P1_P1_0_30min.csv'));
writetable(eventTable, fullfile(outputDir, 'Fig5H_spindle_events_6P1_P1_0_30min.csv'));
writetable(scoreOut, fullfile(outputDir, 'Fig5H_scores_6P1_P1_0_30min.csv'));

fprintf('Exported %d raw EEG samples, %d score epochs, and %d spindle events.\n', ...
    height(traceTable), height(scoreOut), height(eventTable));
fprintf('Sampling rate: %.3f Hz; representative window: %.1f-%.1f min.\n', fs, windowStartMin, windowEndMin);
