function header = make_feature_header()
    % Generate feature names
    sensors = {'ax','ay','az','amag','gx','gy','gz','gmag'};
    base = {};
    
    % Time-domain stats
    for i=1:length(sensors)
        s = sensors{i};
        base = [base, ...
            sprintf('%s_mean',s), sprintf('%s_std',s), sprintf('%s_var',s), ...
            sprintf('%s_rms',s), sprintf('%s_skew',s), sprintf('%s_kurt',s), ...
            sprintf('%s_iqr',s), sprintf('%s_median',s), sprintf('%s_range',s)];
    end
    
    % Energy and zero-crossing
    for i=1:length(sensors)
        s = sensors{i};
        base = [base, sprintf('%s_energy',s), sprintf('%s_zcr',s)];
    end
    
    % Frequency-domain features (from acceleration magnitude)
    base = [base, 'amag_domFreq','amag_centroid','amag_bandwidth', ...
                  'amag_spectralEntropy','amag_autocorrLag'];
    
    header = base;
end