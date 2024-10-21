

function results = MFBAFCM(options)
    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [batCenters, ~] = batAlgorithm(options);

    % Step 3: Run FCM with the initial cluster centers found by MBA
    %[finalCenters, U] = customFCM(options.dataPoints, options.nClusters, options, optimalCenters);
    %opt = fcmOptions(NumClusters = options.nClusters);
       opt = fcmOptions(NumClusters = options.nClusters,...
        ClusterCenters=batCenters');
    % % 
    % opt = fcmOptions(MinImprovement = options.epsilon,...
    % Exponent = options.m,...
    % ClusterCenters = batCenters',...
    %  NumClusters = 'auto',...
    % DistanceMetric = options.DistanceMetric,...
    % MaxNumIteration=options.fcmIterMax);
    disp('FCM Options')
    disp(opt)
%    NumClusters = 'auto',...
    [centers,U,objFcn,info] = fcm(options.dataPoints, opt);

     results = struct();
     results.U = U;
     results.batCenters = batCenters;
     results.centers = centers;
     results.objFcn = objFcn;
     results.info = info;
end

