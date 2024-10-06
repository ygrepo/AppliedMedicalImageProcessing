
function results = computeFCM(options)

    opt = fcmOptions(MinImprovement = options.epsilon,...
    Exponent = options.m,...
    NumClusters = 'auto',...
    DistanceMetric = options.DistanceMetric,...
    MaxNumIteration=options.fcmIterMax);
    [centers,U,objFcn,info] = fcm(options.dataPoints, opt);

     results = struct();
     results.U = U;
     results.centers = centers;
     results.objFcn = objFcn;
     results.info = info;
end

