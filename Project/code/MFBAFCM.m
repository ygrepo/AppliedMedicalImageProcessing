

function U = MFBAFCM(options)
    options.dim = options.nClusters; % Number of clusters is the dimension

    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [optimalCenters, ~] = batAlgorithm(options);

    % Step 3: Run FCM with the initial cluster centers found by MBA
    %[finalCenters, U] = customFCM(options.dataPoints, options.nClusters, options, optimalCenters);

    opt = fcmOptions(MinImprovement = options.epsilon,...
    Exponent = options.m,...
    ClusterCenters = optimalCenters',...
    NumClusters = 'auto',...
    DistanceMetric = options.DistanceMetric,...
    MaxNumIteration=options.fcmIterMax);
    [~, U] = fcm(options.dataPoints, opt);
%    NumClusters=,...
end

