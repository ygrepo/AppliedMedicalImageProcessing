
function segmented_image = computeFCM(image, options)

    opt = fcmOptions(MinImprovement = options.epsilon,...
    Exponent = options.Exponent,...
    NumClusters = 'auto',...
    DistanceMetric = options.DistanceMetric,...
    MaxNumIteration=options.fcmIterMax);
    [centers,U,objFcn,info] = fcm(options.dataPoints, opt);

    % Step 4: Reshape the membership matrix U to form the segmented image
    [~, maxU] = max(U); % Find the cluster with the highest membership for each pixel
    segmented_image = reshape(maxU, size(image));
     results = struct();
     results.
end

