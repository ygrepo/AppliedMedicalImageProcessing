

function results = MFBAFCM(options)
    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [batCenters, ~] = batAlgorithm(options);

    % Step 3: Run FCM with the initial cluster centers found by MBA
    %opt = fcmOptions(NumClusters = options.nClusters);
   % opt = fcmOptions(NumClusters = options.nClusters,...
   %  ClusterCenters=batCenters');
    % % 
    opt = fcmOptions(MinImprovement = options.epsilon,...
    Exponent = options.m,...
    ClusterCenters = batCenters',...
     NumClusters = options.nClusters,...
    MaxNumIteration=options.fcmIterMax);
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


function PC = calculatePartitionCoefficient(U)
    % U: N x c matrix of membership values for each data point in each cluster
    
    % Get the number of data points (N)
    N = size(U, 1);
    
    % Calculate the partition coefficient
    PC = sum(U.^2, 'all') / N;
end
