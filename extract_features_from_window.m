function f = extract_features_from_window(win, fs)
    % Extract features from accelerometer and gyroscope data
    ax = win.ax(:); ay = win.ay(:); az = win.az(:); 
    amag = win.amag(:);
    gx = win.gx(:); gy = win.gy(:); gz = win.gz(:); 
    gmag = win.gmag(:);
    
    N = length(ax);
    if N < 2
        f = nan(1, 0);
        return;
    end
    feats = [];

    % Time-domain stats for accelerometer (ax, ay, az, amag)
    for v = {ax, ay, az, amag}
        vv = v{1};
        feats = [feats, mean(vv), std(vv), var(vv), rms(vv), skewness(vv), ...
                 kurtosis(vv), iqr(vv), median(vv), max(vv)-min(vv)];
    end
    
    % Time-domain stats for gyroscope (gx, gy, gz, gmag)
    for v = {gx, gy, gz, gmag}
        vv = v{1};
        feats = [feats, mean(vv), std(vv), var(vv), rms(vv), skewness(vv), ...
                 kurtosis(vv), iqr(vv), median(vv), max(vv)-min(vv)];
    end

    % Energy and zero-crossing rate for all signals
    for v = {ax, ay, az, amag, gx, gy, gz, gmag}
        vv = v{1};
        energy = sum(vv.^2) / max(N,1);
        zc = sum(vv(1:end-1).*vv(2:end) < 0);
        feats = [feats, energy, zc];
    end

    % Frequency-domain features from acceleration magnitude
    Y = fft(amag);
    P2 = abs(Y / N).^2;
    P1 = P2(1:floor(N/2));
    freqs = (0:length(P1)-1)' * (fs / N);

    if sum(P1) <= eps
        pnorm = ones(size(P1)) / numel(P1);
    else
        pnorm = P1 / sum(P1);
    end

    centroid = sum(freqs .* pnorm);
    bandwidth = sqrt(sum(((freqs - centroid).^2) .* pnorm));
    specEntropy = -sum(pnorm .* log2(pnorm + eps));
    [~, idxmax] = max(P1);
    domFreq = freqs(max(1, idxmax));

    % Autocorrelation lag
    acf = xcorr(amag - mean(amag), 'coeff');
    mid = ceil(length(acf)/2);
    acf_pos = acf(mid+1:end);
    if isempty(acf_pos)
        lag = 0;
    else
        [pks, locs] = findpeaks(acf_pos);
        if isempty(locs)
            lag = 0;
        else
            lag = locs(1);
        end
    end

    feats = [feats, domFreq, centroid, bandwidth, specEntropy, lag];
    f = feats;
end