clear; close all; clc;
 

rng(0); % reproducible

%%  Read data 

dataFolder = 'Dataset'; % data folder
fs_expected = 32; % expected sampling rate
useResample = true;   % resample signals to fs_expected if timestamps available

csvFiles = dir(fullfile(dataFolder,'*.csv'));
if isempty(csvFiles)
    error('No CSV files found in %s.', dataFolder);
end

rawData = struct();

for k = 1:length(csvFiles)
    fname = fullfile(csvFiles(k).folder, csvFiles(k).name);
    fprintf("Loading %s\n", csvFiles(k).name);

    % Read the CSV file
    try
        M = readmatrix(fname);
    catch ME
        warning('Could not read %s: %s', csvFiles(k).name, ME.message);
        continue;
    end
    
    % Validate columns
    if size(M,2) ~= 7
        warning("Expected 7 columns in %s, found %d. Skipping.", csvFiles(k).name, size(M,2));
        continue;
    end

    % Extract columns: label, ax, ay, az, gx, gy, gz
    label = M(:,1);
    ax = M(:,2); 
    ay = M(:,3); 
    az = M(:,4);
    gx = M(:,5); 
    gy = M(:,6); 
    gz = M(:,7);

    % Remove any NaN or Inf values
    validIdx = ~any(isnan(M) | isinf(M), 2);
    if sum(~validIdx) > 0
        fprintf('   Removing %d invalid rows\n', sum(~validIdx));
        label = label(validIdx);
        ax = ax(validIdx);
        ay = ay(validIdx);
        az = az(validIdx);
        gx = gx(validIdx);
        gy = gy(validIdx);
        gz = gz(validIdx);
    end
    
    N = length(ax);
    if N < 100
        warning('File %s has too few samples (%d), skipping.', csvFiles(k).name, N);
        continue;
    end

    % Estimate sampling rate from label transitions
   
    fs_estimated = fs_expected; % default fallback
    
    labelDiff = diff(label);
    transitions = find(labelDiff ~= 0);
    
    if length(transitions) >= 2
        % Calculate samples between transitions
        segmentLengths = diff([1; transitions; N]);
        
        % Remove outliers (segments that are too short or too long)
        medianLen = median(segmentLengths);
        validSegs = segmentLengths(segmentLengths > medianLen*0.5 & segmentLengths < medianLen*2);
        
        if ~isempty(validSegs)
            avgSegmentLength = mean(validSegs);
            
           
            
            % Alternative: Use spectral analysis on acceleration magnitude
            amag_raw = sqrt(ax.^2 + ay.^2 + az.^2);
            
            % Detrend for frequency analysis
            amag_detrend = detrend(amag_raw);
            
            % FFT to find dominant frequency (gait frequency)
            NFFT = 2^nextpow2(N);
            Y = fft(amag_detrend, NFFT);
            P = abs(Y(1:NFFT/2+1)).^2;
            
            % Expected gait frequency: 0.5-2.5 Hz (normal walking)
            
            test_fs_values = [20, 25, 30, 32, 35, 40, 50];
            best_fs = fs_expected;
            best_score = 0;
            
            for test_fs = test_fs_values
                freq_axis = test_fs * (0:(NFFT/2)) / NFFT;
                
                % Find peak in gait frequency range (0.5-2.5 Hz)
                valid_range = freq_axis >= 0.5 & freq_axis <= 2.5;
                if any(valid_range)
                    [peak_val, ~] = max(P(valid_range));
                    
                    % Score based on peak magnitude and proximity to expected fs
                    fs_proximity = 1 / (1 + abs(test_fs - fs_expected));
                    score = peak_val * fs_proximity;
                    
                    if score > best_score
                        best_score = score;
                        best_fs = test_fs;
                    end
                end
            end
            
            fs_estimated = best_fs;
            fprintf('   Estimated fs = %.2f Hz (spectral analysis)\n', fs_estimated);
        else
            % Fallback: assume 30-32 Hz 
            fs_estimated = fs_expected;
            fprintf('   Using default fs = %.2f Hz (insufficient label transitions)\n', fs_estimated);
        end
    else
        % Estimate from total samples assuming 6 min
        
        session_duration = 360; % seconds
        fs_estimated = N / session_duration;
        
        % Clamp to reasonable range
        if fs_estimated < 20 || fs_estimated > 50
            fprintf('   Calculated fs = %.2f Hz seems unrealistic, using default %.2f Hz\n', ...
                    fs_estimated, fs_expected);
            fs_estimated = fs_expected;
        else
            fprintf('   Estimated fs = %.2f Hz (from sample count / 6 min)\n', fs_estimated);
        end
    end
    
    % Build synthetic time vector
    t = (0:N-1)' / fs_estimated;

    % Store data
    rawData(k).name = csvFiles(k).name;
    rawData(k).label = label;
    rawData(k).ax = ax;
    rawData(k).ay = ay;
    rawData(k).az = az;
    rawData(k).gx = gx;
    rawData(k).gy = gy;
    rawData(k).gz = gz;
    rawData(k).t = t;
    rawData(k).fs = fs_estimated;
    rawData(k).numSamples = N;

    fprintf('   Loaded %d samples, estimated fs = %.2f Hz\n', N, fs_estimated);
end

% Remove empty entries
rawData = rawData(~arrayfun(@(x) isempty(x.name), rawData));

if isempty(rawData)
    error('No valid data files loaded. Check your data folder and file format.');
end

fprintf('\nSuccessfully loaded %d files\n\n', length(rawData));

%%  Preprocessing: detrend, lowpass, magnitude, normalize
fprintf('Preprocessing signals...\n');
preprocData = struct();
for k=1:numel(rawData)
    s.ax = rawData(k).ax;
    s.ay = rawData(k).ay;
    s.az = rawData(k).az;
    s.gx = rawData(k).gx;
    s.gy = rawData(k).gy;
    s.gz = rawData(k).gz;
    fs = rawData(k).fs;

    % Detrend accelerometer
    s.ax = detrend(s.ax);
    s.ay = detrend(s.ay);
    s.az = detrend(s.az);
    
    % Detrend gyroscope
    s.gx = detrend(s.gx);
    s.gy = detrend(s.gy);
    s.gz = detrend(s.gz);

    % Low-pass filter (cutoff 10 Hz for gait)
    fc = 10;
    if fs <= 2*fc
        warning('Sampling rate (%.2f Hz) too low for fc=%.1fHz; skipping lowpass filter for file %s', fs, fc, rawData(k).name);
    else
        [b,a] = butter(4, fc/(fs/2), 'low');
        s.ax = filtfilt(b,a,s.ax);
        s.ay = filtfilt(b,a,s.ay);
        s.az = filtfilt(b,a,s.az);
        s.gx = filtfilt(b,a,s.gx);
        s.gy = filtfilt(b,a,s.gy);
        s.gz = filtfilt(b,a,s.gz);
    end

    % Compute magnitudes
    s.amag = sqrt(s.ax.^2 + s.ay.^2 + s.az.^2);
    s.gmag = sqrt(s.gx.^2 + s.gy.^2 + s.gz.^2);

    % Normalize (z-score) - guard against zero std
    ax_mean = mean(s.ax); ax_std = std(s.ax); if ax_std==0, ax_std=1; end
    ay_mean = mean(s.ay); ay_std = std(s.ay); if ay_std==0, ay_std=1; end
    az_mean = mean(s.az); az_std = std(s.az); if az_std==0, az_std=1; end
    am_mean = mean(s.amag); am_std = std(s.amag); if am_std==0, am_std=1; end
    
    gx_mean = mean(s.gx); gx_std = std(s.gx); if gx_std==0, gx_std=1; end
    gy_mean = mean(s.gy); gy_std = std(s.gy); if gy_std==0, gy_std=1; end
    gz_mean = mean(s.gz); gz_std = std(s.gz); if gz_std==0, gz_std=1; end
    gm_mean = mean(s.gmag); gm_std = std(s.gmag); if gm_std==0, gm_std=1; end

    s.ax = (s.ax - ax_mean) ./ ax_std;
    s.ay = (s.ay - ay_mean) ./ ay_std;
    s.az = (s.az - az_mean) ./ az_std;
    s.amag = (s.amag - am_mean) ./ am_std;
    
    s.gx = (s.gx - gx_mean) ./ gx_std;
    s.gy = (s.gy - gy_mean) ./ gy_std;
    s.gz = (s.gz - gz_mean) ./ gz_std;
    s.gmag = (s.gmag - gm_mean) ./ gm_std;

    % Store
    preprocData(k).name = rawData(k).name;
    preprocData(k).s = s;
    preprocData(k).fs = fs;
end

%%  Segmentation: sliding window - SEPARATE FD and MD

windowSec = 2.0; % window length
overlapSec = 1.0; % overlap between windows

fprintf('Windowing (%.1fs windows, %.1fs overlap)...\n', windowSec, overlapSec);
fs_use = fs_expected;
winSamples = round(windowSec * fs_use);
stepSamples = max(1, round((windowSec - overlapSec) * fs_use));

% Initialize separate containers for FD (train) and MD (test)
trainFeatures = [];
trainLabels = [];
trainFileIdx = [];

testFeatures = [];
testLabels = [];
testFileIdx = [];

for k=1:numel(preprocData)
    s = preprocData(k).s;
    N = length(s.ax);
    if N < winSamples
        continue; % skip too-short files
    end
    
    % Determine if this is FD (train) or MD (test)
    filename = preprocData(k).name;
    isFD = contains(filename, '_FD.csv');
    isMD = contains(filename, '_MD.csv');
    
    if ~isFD && ~isMD
        warning('File %s does not match FD or MD pattern, skipping.', filename);
        continue;
    end
    
    idx = 1:stepSamples:(N - winSamples + 1);
    for i=1:length(idx)
        i0 = idx(i);
        i1 = i0 + winSamples - 1;
        win.ax = s.ax(i0:i1);
        win.ay = s.ay(i0:i1);
        win.az = s.az(i0:i1);
        win.amag = s.amag(i0:i1);
        win.gx = s.gx(i0:i1);
        win.gy = s.gy(i0:i1);
        win.gz = s.gz(i0:i1);
        win.gmag = s.gmag(i0:i1);
        
        % Extract features
        f = extract_features_from_window(win, fs_use);
        
        % Add to appropriate set
        if isFD
            trainFeatures = [trainFeatures; f(:)'];
            trainLabels = [trainLabels; k];
            trainFileIdx = [trainFileIdx; k];
        else % isMD
            testFeatures = [testFeatures; f(:)'];
            testLabels = [testLabels; k];
            testFileIdx = [testFileIdx; k];
        end
    end
end

% Build header
featureHeader = make_feature_header();

fprintf('Extracted %d TRAIN windows (FD) and %d TEST windows (MD)\n', ...
    size(trainFeatures,1), size(testFeatures,1));
fprintf('Total features per window: %d\n', size(trainFeatures,2));

if isempty(trainFeatures) || isempty(testFeatures)
    error('Missing FD or MD data. Check that files follow naming pattern U<N>NW_FD.csv and U<N>NW_MD.csv');
end

%%  Map files to user IDs 
userMap = zeros(length(rawData),1);
for k=1:numel(rawData)
    name = rawData(k).name;
    tok = regexp(name,'U(\d{1,2})','tokens','once');
    if ~isempty(tok)
        userMap(k) = str2double(tok{1});
    else
        userMap(k) = k; % fallback: unique per file
    end
end

% Map each window label (file index) to user id
trainUserIDs = zeros(size(trainFileIdx));
for i=1:length(trainFileIdx)
    trainUserIDs(i) = userMap(trainFileIdx(i));
end

testUserIDs = zeros(size(testFileIdx));
for i=1:length(testFileIdx)
    testUserIDs(i) = userMap(testFileIdx(i));
end

uniqueUsers = unique(trainUserIDs);
numUsers = length(uniqueUsers);
fprintf('Detected %d unique users in training set.\n', numUsers);

%%  Feature Variability Analysis across users 
fprintf('Analyzing feature variability across users (training data)...\n');
X = trainFeatures;
Y = trainUserIDs;

% Select a few important features to visualize
featToPlot = [1, 10, 20, 30, 40]; % Adjust based on total features
featToPlot = featToPlot(featToPlot <= size(X,2));

figure('Name','Feature Variability Across Users (Training Data)');
for i = 1:length(featToPlot)
    subplot(2,3,i);
    featIdx = featToPlot(i);
    boxplot(X(:,featIdx), Y);
    title(sprintf('Feature: %s', featureHeader{featIdx}));
    xlabel('User ID');
    ylabel('Feature Value');
    grid on;
end
sgtitle('Feature Variability Analysis - Training Set (FD)');

%%  Feature selection (mRMR) or PCA fallback

numFeaturesSelected = 30;  % number of features
doFeatureSelection = true; % mRMR selection

if doFeatureSelection
    fprintf('Running feature selection on training data (mRMR if available, otherwise PCA)\n');
    try
        % fscmrmr requires Statistics and Machine Learning Toolbox
        [idxfs, mrmrScoresRaw] = fscmrmr(trainFeatures, trainUserIDs);

        % Normalize scores 0–1
        mrmrScores = mrmrScoresRaw / max(mrmrScoresRaw);

        % Sort features by score descending
        [sortedScores, sortIdx] = sort(mrmrScores, 'descend');
        sortedFeatures = featureHeader(sortIdx);

        % Plot only the top N features for readability
        topN = min(20, length(mrmrScores));

        figure('Name','Feature Importance (mRMR)', 'Position',[200 200 900 500]);
        bar(sortedScores(1:topN));
        title('Feature Importance (Top Ranked by mRMR)');
        ylabel('Normalized Importance');
        grid on;

        set(gca, 'XTick', 1:topN, ...
                 'XTickLabel', sortedFeatures(1:topN), ...
                 'XTickLabelRotation', 45);

        % Selected features
        selN = min(numFeaturesSelected, size(trainFeatures,2));
        selIdx = idxfs(1:selN);
        XtrainSel = trainFeatures(:, selIdx);
        XtestSel = testFeatures(:, selIdx);
        selectedHeader = featureHeader(selIdx);

        fprintf('Selected top %d features by mRMR.\n', length(selIdx));

    catch ME
        fprintf('mRMR not available or failed: %s\nUsing PCA to reduce features.\n', ME.message);
        [coeff, score, ~, ~, explained] = pca(trainFeatures);
        Kpca = min(numFeaturesSelected, size(score,2));
        XtrainSel = score(:,1:Kpca);
        XtestSel = (testFeatures - mean(trainFeatures,1)) * coeff(:,1:Kpca);
        selectedHeader = arrayfun(@(i) sprintf('PCA_%d',i), 1:Kpca,'uni',0);
    end
else
    XtrainSel = trainFeatures;
    XtestSel = testFeatures;
    selectedHeader = featureHeader;
end

%%  PCA Projection Scatter Plot 
fprintf('Plotting PCA Projection (2D) - Train and Test...\n');

% Compute PCA on the training features
[coeff, trainScore, ~, ~, explained] = pca(trainFeatures);

% Project test features into same PCA space
testScore = (testFeatures - mean(trainFeatures,1)) * coeff;

% 2D Scatter Plot - Training Data
figure('Name','PCA 2D Projection - Training (FD)');
gscatter(trainScore(:,1), trainScore(:,2), trainUserIDs);
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA Projection - Training Data (First Day)');
legend('Location','best');
grid on;

% 2D Scatter Plot - Test Data
figure('Name','PCA 2D Projection - Test (MD)');
gscatter(testScore(:,1), testScore(:,2), testUserIDs);
xlabel(sprintf('PC1 (%.1f%%)', explained(1)));
ylabel(sprintf('PC2 (%.1f%%)', explained(2)));
title('PCA Projection - Test Data (Second Day)');
legend('Location','best');
grid on;

%%  labels for training and testing
Xtrain = XtrainSel;
Ytrain = trainUserIDs;
Xtest = XtestSel;
Ytest = testUserIDs;

% Map class labels to consecutive indices 1..numClasses
classList = unique(Ytrain);
numClasses = length(classList);
mapToIdx = containers.Map(num2cell(classList), num2cell(1:numClasses));

YtrainMapped = arrayfun(@(v) mapToIdx(v), Ytrain);

% Map test labels
YtestMapped = zeros(size(Ytest));
for i = 1:length(Ytest)
    if isKey(mapToIdx, Ytest(i))
        YtestMapped(i) = mapToIdx(Ytest(i));
    else
        YtestMapped(i) = 0; % Mark as missing
    end
end

missingTest = (YtestMapped==0);
if any(missingTest)
    warning('%d test samples from users not in training set. Those windows will be removed.', sum(missingTest));
    keepIdx = ~missingTest;
    Xtest = Xtest(keepIdx,:);
    Ytest = Ytest(keepIdx);
    YtestMapped = YtestMapped(keepIdx);
end

fprintf('Training on %d users with %d windows from Day 1 (FD)\n', numClasses, size(Xtrain,1));
fprintf('Testing on %d windows from Day 2 (MD)\n', size(Xtest,1));

%%  Normalize features (z-score on training)
mu = mean(Xtrain,1);
sigma = std(Xtrain,[],1);
sigma(sigma==0)=1;
XtrainN = (Xtrain - mu) ./ sigma;
XtestN = (Xtest - mu) ./ sigma;

%%  Train MLP 

hiddenLayerSizes = [64 32]; % MLP architecture

fprintf('Training MLP with layers [%s]\n', sprintf('%d ', hiddenLayerSizes));
net = patternnet(hiddenLayerSizes, 'trainscg');
net.performFcn = 'crossentropy';
net.trainParam.epochs = 200;
net.divideFcn = 'dividetrain';

% Prepare targets: one-hot encoding
Ttrain = zeros(numClasses, size(XtrainN,1));
for i=1:size(XtrainN,1)
    ci = YtrainMapped(i);
    Ttrain(ci, i) = 1;
end

% Train
[net, tr] = train(net, XtrainN', Ttrain);

%%  Learning Curve plot
figure('Name','Learning Curve');
plot(tr.perf, 'LineWidth', 1.5);
xlabel('Epoch'); 
ylabel('Loss (Performance)');
title('Model Learning Curve (Training on Day 1)');
legend('Training Loss');
grid on;

% Save network
save('trainedMLP_FD_MD.mat','net','mu','sigma','selectedHeader','featureHeader','classList');

%%  Evaluate identification 
fprintf('\n=== TESTING ON DAY 2 (MD) ===\n');
Yprob = net(XtestN');
[~, YpredMapped] = max(Yprob,[],1);
YpredMapped = YpredMapped';

% Compute accuracy
acc = sum(YpredMapped == YtestMapped) / numel(YtestMapped);
fprintf('Identification Accuracy (Day 2) = %.2f%%\n', acc*100);

% Confusion matrix
C = confusionmat(YtestMapped, YpredMapped, 'Order', 1:numClasses);
figure;
confusionchart(C, arrayfun(@(c) sprintf('U%d', classList(c)), 1:numClasses, 'UniformOutput', false));
title(sprintf('Confusion Matrix - Day 2 Test (Accuracy %.2f%%)', acc*100));

%%  Softmax probability histograms
fprintf('Plotting Softmax Probability Histograms (Day 2 Test)...\n');
numUsersEval = numClasses;
figure('Name','Softmax Probability Histograms (Day 2)','Position',[100 100 1200 800]);
ncols = 3;
nrows = ceil(numUsersEval/ncols);

for ui = 1:numUsersEval
    genuine = Yprob(ui, YtestMapped == ui);
    impostor = Yprob(ui, YtestMapped ~= ui);
    
    subplot(nrows, ncols, ui);
    hold on;
    if ~isempty(genuine)
        histogram(genuine, 'Normalization','pdf', 'FaceAlpha',0.6);
    end
    if ~isempty(impostor)
        histogram(impostor, 'Normalization','pdf', 'FaceAlpha',0.6);
    end
    title(sprintf('User %d', classList(ui)));
    xlabel('Probability'); 
    ylabel('Density');
    legend('Genuine','Impostor');
    grid on;
end
sgtitle('Softmax Probability Histograms - Day 2 (Genuine vs Impostor)');

%%  Verification Evaluation 
fprintf('Computing FAR / FRR / EER on Day 2 (one-vs-rest)\n');
EERs = zeros(numClasses,1);
AUCs = zeros(numClasses,1);

for u = 1:numClasses
    genuine = Yprob(u, YtestMapped == u);
    impostor = Yprob(u, YtestMapped ~= u);

    labels = [ones(numel(genuine),1); zeros(numel(impostor),1)];
    scores = [genuine(:); impostor(:)];

    if numel(unique(labels)) < 2
        EERs(u) = NaN;
        AUCs(u) = NaN;
        continue;
    end

    [Xroc, Yroc, Troc, AUC] = perfcurve(labels, scores, 1);
    AUCs(u) = AUC;

    FPR = Xroc;
    TPR = Yroc;
    FRR = 1 - TPR;
    diffVals = abs(FPR - FRR);
    [~, idxEER] = min(diffVals);
    EER = (FPR(idxEER) + FRR(idxEER)) / 2;
    EERs(u) = EER;

    fprintf('User %d -> AUC = %.3f, EER = %.3f\n', classList(u), AUC, EER);
end

meanEER = nanmean(EERs);
meanAUC = nanmean(AUCs);
fprintf('\n=== FINAL RESULTS (Day 2 Testing) ===\n');
fprintf('Mean EER across users = %.3f\n', meanEER);
fprintf('Mean AUC across users = %.3f\n', meanAUC);
fprintf('Identification Accuracy = %.2f%%\n', acc*100);

%%  Save results
results.AUCs = AUCs;
results.EERs = EERs;
results.meanEER = meanEER;
results.meanAUC = meanAUC;
results.accuracy = acc;
results.confusion = C;
results.classList = classList;
results.net = net;
results.testMethod = 'Train_FD_Test_MD';
save('results_FD_MD.mat','results');

fprintf('\n=== Pipeline Complete ===\n');
fprintf('Training: Day 1 (FD files), Testing: Day 2 (MD files)\n');
fprintf('Results saved to: trainedMLP_FD_MD.mat, results_FD_MD.mat\n');
