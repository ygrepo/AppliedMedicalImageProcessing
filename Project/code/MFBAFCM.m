

function segmented_image = MFBAFCM(image, options)
    options.dim = options.nClusters; % Number of clusters is the dimension

    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [optimalCenters, ~] = batAlgorithm(options);

    % Step 3: Run FCM with the initial cluster centers found by MBA
    %[finalCenters, U] = customFCM(options.dataPoints, options.nClusters, options, optimalCenters);

    opt = fcmOptions(MinImprovement = options.epsilon,...
    Exponent = options.Exponent,...
    ClusterCenters = optimalCenters',...
    NumClusters = 'auto',...
    DistanceMetric = options.DistanceMetric,...
    MaxNumIteration=options.fcmIterMax);
    [~, U] = fcm(options.dataPoints, opt);
%    NumClusters=,...

    % Step 4: Reshape the membership matrix U to form the segmented image
    [~, maxU] = max(U); % Find the cluster with the highest membership for each pixel
    segmented_image = reshape(maxU, size(image));
end

