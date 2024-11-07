% Load an example MRI image
clearvars 
clc
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Select the 15th slice of the MRI
slice_number = 15;
sliceData = D(:,:,slice_number);  % uint8 data

% Normalize the image data
data = double(sliceData) / 255;  % Normalize pixel values to [0, 1]

nClusters = 4;
fcmOpt = fcmOptions(NumClusters = nClusters);
disp('FCM Options')
disp(fcmOpt)

% Convert the 2D image into a 2D coordinate + intensity format
[H, W] = size(data);            % Get the dimensions of the image
[X, Y] = meshgrid(1:W, 1:H);    % Generate (x, y) coordinates
dataPoints = [X(:), Y(:), data(:)];  % Flatten `data` to match the coordinates

% Apply Fuzzy C-Means (FCM) clustering
[fcmCenters, fcmU] = fcm(dataPoints, fcmOpt);

% Get the cluster membership for each pixel
[~, maxU] = max(fcmU);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segFCMImg = reshape(maxU, H, W);

% Display the segmented image
imshow(segFCMImg, []);
title('Segmented Image using FCM');

% 
% % Options for FCM and Bat Algorithm
% options = struct();
% options.NumClusters = nClusters;
% options.ClusterCenters = [];
% options.Exponent = 2;
% options.MaxNumIteration = 300;
% options.DistanceMetric = 'euclidean';
% options.MinImprovement = 1e-6;
% options.Verbose = 1;
% options.ClusterVolume = 1;
% options.Exponent = 2;
% options.dataPoints = data;  % Original image data for FCM
% options.lambda = 0;
% 
% % BAT Options for 2D image clustering
% options.nBats = 50;
% options.BATIterMax = 100;
% options.lowerBound = [1, 1];  % 2D bounds for x, y coordinates
% options.upperBound = [size(data, 1), size(data, 2)];
% options.Qmin = 0;
% options.Qmax = 2;
% options.loudness = 0.5;  % Initial loudness
% options.loudnessCoefficient = 0.9;
% options.pulseRate = 0.5;  % Initial pulse rate
% options.gamma = 0.95;  % Decay rate for pulse rate
% options.MinNumIteration = 50;
% options.UsePerturbation = true;
% options.PerturbationFactor = 0.01;
% options.image = data;  % Pass the 2D image data to batAlgorithm2D
% 
% % Apply BAT + FCM clustering
% segImgInresults = MFBAFCM(options);
% 
% % Get the cluster membership for each pixel
% [~, maxU] = max(segImgInresults.U, [], 1);  % Determine the cluster with the highest membership for each pixel
% 
% % Reshape the result back to the original image size
% segBATFCMImg = reshape(maxU, size(sliceData));
% 
% % Display the segmented image
% imshow(segBATFCMImg, []);
% title('Segmented Image using BAT + FCM');


%% Function for Modified BAT + FCM (MFBAFCM)
function results = MFBAFCM(options)
    % Step 1: Run the Modified Bat Algorithm (MBA) to find initial cluster centers in 2D
    [batCenters, ~] = batAlgorithm2D(options);

    % Convert the 2D image into a 2D coordinate + intensity format if needed
    [H, W] = size(options.image);  % Assume options.image contains the grayscale image
    [X, Y] = meshgrid(1:W, 1:H);   % Generate (x, y) coordinates
    dataPoints = [X(:), Y(:), double(options.image(:)) / 255];  % Convert image to [x, y, intensity]

    % Match batCenters to the dataPoints format, converting to [x, y, intensity]
    intensityValues = arrayfun(@(i) options.image(round(batCenters(i, 1)), round(batCenters(i, 2))), ...
                               1:size(batCenters, 1))';
    batCenters = [batCenters, intensityValues];  % Convert batCenters to [x, y, intensity]

    % Set FCM options with bat-generated cluster centers
    opt = fcmOptions(MinImprovement = options.MinImprovement, ...
                     Exponent = options.Exponent, ...
                     ClusterCenters = batCenters, ...
                     NumClusters = options.NumClusters, ...
                     MaxNumIteration = options.MaxNumIteration);

    % Display FCM options for debugging
    disp('FCM Options')
    disp(opt)

    % Run FCM with the data points in [x, y, intensity] format
    [centers, U, objFcn, info] = fcm(dataPoints, opt);

    % Store results in a structured format
    results = struct();
    results.U = U;
    results.batCenters = batCenters;
    results.centers = centers;
    results.objFcn = objFcn;
    results.info = info;
end


% function results = MFBAFCM(options)
%     % Step 1: Run the Modified Bat Algorithm (MBA) to find initial cluster centers in 2D
%     [batCenters, ~] = batAlgorithm2D(options);
% 
%     opt = fcmOptions(MinImprovement = options.MinImprovement,...
%     Exponent = options.Exponent,...
%     ClusterCenters = batCenters,...
%      NumClusters = options.NumClusters,...
%     MaxNumIteration=options.MaxNumIteration);
%     disp('FCM Options')
%     disp(opt)
% %    NumClusters = 'auto',...
%     [centers,U,objFcn,info] = fcm(options.dataPoints, opt);
% 
%     % % Step 2: Prepare initial cluster centers for FCM
%     % options.ClusterCenters = batCenters;  % Use bat-generated centers
%     % 
%     % % Step 3: Run FCM with the initial cluster centers found by the bat algorithm
%     % disp('Running Fuzzy C-Means (FCM) with initial centers from Bat Algorithm');
%     % [centers, U, objFcn, info] = customFCM(options.dataPoints(:), options);
% 
%     % Store results in a structured format
%     results = struct();
%     results.U = U;
%     results.batCenters = batCenters;
%     results.centers = centers;
%     results.objFcn = objFcn;
%     results.info = info;
% end
function [bestSol, bestFitness] = batAlgorithm2D(options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rates
    loudnesses = options.loudness * ones(options.nBats, 1);  % Per-bat loudness

    % Initialize bat positions for 2D cluster centers
    bats = cat(3, ...
               randi([1, size(options.image, 1)], options.nBats, options.NumClusters), ... % X-coordinates
               randi([1, size(options.image, 2)], options.nBats, options.NumClusters));    % Y-coordinates
    velocities = zeros(options.nBats, options.NumClusters, 2);  % 2D velocities for each cluster per bat
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats
    for i = 1:options.nBats
        fitness(i) = calculateFitness2D(squeeze(bats(i, :, :)), options.image);  % 2D fitness based on image
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestSol = squeeze(bats(idx, :, :));  % Initial best solution (NumClusters x 2)

    % Main loop for the BAT algorithm
    for t = 1:options.BATIterMax
        for i = 1:options.nBats
            % Update frequency, velocity, and position in 2D for each cluster
            Q = options.Qmin + (options.Qmax - options.Qmin) * rand;
            velocities(i, :, :) = squeeze(velocities(i, :, :)) + (squeeze(bats(i, :, :)) - bestSol) * Q;
            % velocities(i, :, :) = velocities(i, :, :) + (squeeze(bats(i, :, :)) - bestSol) * Q;
            newSolution = squeeze(bats(i, :, :)) + squeeze(velocities(i, :, :));
            % Enforce boundary constraints for each cluster center in 2D
            newSolution = enforceBoundaries2D(newSolution, size(options.image));

            % Local search
            if rand > pulseRates(i)
                newSolution = bestSol + 0.01 * randn(options.NumClusters, 2);  % Small random perturbation in 2D
            end

            % Evaluate the new solution's fitness
            newFitness = calculateFitness2D(newSolution, options.image);

            % Acceptance criteria
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                bats(i, :, :) = newSolution;
                fitness(i) = newFitness;
                loudnesses(i) = options.loudnessCoefficient * loudnesses(i);  % Decrease loudness
                pulseRates(i) = pulseRates(i) * (1 - exp(-options.gamma * t));  % Increase pulse rate
            end

            % Update global best
            if newFitness < bestFitness
                bestSol = newSolution;
                bestFitness = newFitness;
            end
        end
    end
end

% Function to enforce boundary constraints in 2D
function newSolution = enforceBoundaries2D(newSolution, imgSize)
    % Ensure each cluster center's position remains within image bounds
    newSolution(:, 1) = max(1, min(imgSize(1), newSolution(:, 1)));  % X-bound for each cluster center
    newSolution(:, 2) = max(1, min(imgSize(2), newSolution(:, 2)));  % Y-bound for each cluster center
end

% Example fitness function for 2D image processing
function fitness = calculateFitness2D(centers, image)
    % Centers is an Nx2 array, where each row is a [x, y] coordinate of a cluster center
    fitness = 0;
    for j = 1:size(centers, 1)
        x = round(centers(j, 1));
        y = round(centers(j, 2));

        % Ensure (x, y) are within the image boundaries
        x = max(1, min(size(image, 1), x));
        y = max(1, min(size(image, 2), y));

        % Fitness based on the pixel intensity at each center position
        fitness = fitness - double(image(x, y));  % Higher intensity preferred (negative for maximization)
    end
end

%%

function intraCluster = calculateIntraCluster2D(dataPoints, clusterCenters)
    % Ensure dataPoints and clusterCenters have compatible dimensions
    if size(dataPoints, 2) ~= size(clusterCenters, 2)
        error('Data points and cluster centers must have the same number of columns.');
    end

    % Compute pairwise squared Euclidean distances between data points and cluster centers
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');
    
    % Sum all distances and average by the number of data points
    intraCluster = sum(distances, 'all') / size(dataPoints, 1);
end

function SC = calculateSpatialCoherence(U, dataPoints, clusterCenters, image, m)
    % U: N x C matrix of membership values for each pixel
    % dataPoints: N x D matrix of pixel values
    % clusterCenters: C x D matrix of cluster centers (intensity or color values)
    % image: H x W matrix for grayscale or H x W x 3 for RGB
    % m: Fuzziness exponent

    % Calculate intra-cluster distances (pixel-cluster center similarity)
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');

    % Weight distances by membership values raised to m
    numerator = sum((U.^m) .* distances, 1);

    % Calculate neighborhood coherence within clusters
    [H, W, ~] = size(image);
    coherence = 0;
    for i = 1:size(U, 2)  % For each cluster
        % Approximate coherence by neighboring similarity
        coherence = coherence + neighborhoodSimilarity(image, U(:, i), clusterCenters(i, :), H, W);
    end

    % Sum coherence with intra-cluster similarity as the partition index
    SC = sum(numerator) + coherence;
end

function coherence = neighborhoodSimilarity(image, U_col, clusterCenter, H, W)
    % U_col: 1D vector of membership values for each pixel to a single cluster
    % clusterCenter: Intensity or color of the cluster center
    % image: H x W or H x W x 3 matrix
    % H, W: Height and width of the image
    % Neighborhood coherence based on local pixel intensity similarity
    
    coherence = 0;
    for x = 2:H-1
        for y = 2:W-1
            % Local 3x3 neighborhood around the pixel
            local_patch = image(x-1:x+1, y-1:y+1);
            % Difference between cluster center and neighborhood
            coherence = coherence + sum((local_patch - clusterCenter).^2, 'all') * U_col((x-1)*W + y);
        end
    end
    coherence = coherence / (H * W); % Normalize by image size
end

function PC = calculatePartitionCoefficient(U)
    % U: N x C matrix of membership values for each pixel to clusters
    
    % Get the number of data points (pixels)
    N = size(U, 1);
    
    % Calculate the partition coefficient
    PC = sum(U.^2, 'all') / N;
end

%%
function varargout = customFCM(data, options)
    % Initialize variables
    dataSize = size(data, 1);
    objFcn = zeros(options.MaxNumIteration, 1);
    maxAutoK = 10;
    
    % Determine the cluster numbers based on options
    if isequal(options.NumClusters, fuzzy.clustering.FCMOptions.DefaultNumClusters)
        kStart = 2;
        maxNumClusters = min(kStart + maxAutoK - 1, size(data, 1) - 1);
        kValues = kStart:maxNumClusters;
    else
        kValues = options.NumClusters;
    end
    
    lastResults = [];
    centers = options.ClusterCenters;
    validityIndexOutput = zeros(numel(kValues), 2);
    info = initializeInfoStructure(numel(kValues));
    minKIndex = Inf;
    
    for k = kValues
        options.ClusterCenters = [];
        options.NumClusters = k;
        
        % Initialize fuzzy partition matrix
        fuzzyPartMat = fuzzy.clustering.initfcm(options, dataSize);
        
        for iterId = 1:options.MaxNumIteration
            % Perform one step of FCM with spatially-aware objective
            [center, fuzzyPartMat, objFcn(iterId), covMat, stepBrkCond] = stepfcm2D(data, fuzzyPartMat, options);
            
            % Check for termination
            brkCond = checkBreakCondition(options, objFcn(iterId), iterId, stepBrkCond);
            if brkCond.isTrue, objFcn(iterId+1:end) = []; break; end
        end
        
        % Calculate cluster validity index (e.g., Dunn or DB index)
        validityIndex = fuzzy.clustering.clusterValidityIndex(data, center, fuzzyPartMat);
        
        % Track the optimal cluster number and save results
        if isempty(lastResults) || validityIndex < lastResults.validityIndex
            lastResults = saveOptimalResults(center, fuzzyPartMat, objFcn, covMat, validityIndex, k);
            minKIndex = k;
        end
        
        % Save info for this cluster count
        info = saveClusterInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k);
    end
    
    % Assign output variables
    [varargout{1:nargout}] = assignOutputs(info, minKIndex);
end

function info = initializeInfoStructure(numFolds)
    info = struct('NumClusters', zeros(1, numFolds), ...
                  'ClusterCenters', cell(1, numFolds), ...
                  'FuzzyPartitionMatrix', cell(1, numFolds), ...
                  'ObjectiveFcnValue', cell(1, numFolds), ...
                  'CovarianceMatrix', cell(1, numFolds), ...
                  'ValidityIndex', zeros(1, numFolds), ...
                  'OptimalNumClusters', 0);
end

function [center, newFuzzyPartMat, objFcn, covMat, brkCond] = stepfcm2D(data, fuzzyPartMat, options)
    numCluster = options.NumClusters;
    expo = options.Exponent;
    
    % Modify Fuzzy Partition Matrix
    memFcnMat = fuzzyPartMat.^expo;
    
    % Calculate new cluster centers
    center = memFcnMat * data ./ (sum(memFcnMat, 2) * ones(1, size(data, 2)));
    
    % Calculate Euclidean or other specified distances
    dist = fuzzy.clustering.euclideandist(center, data); % Adapt this for 2D image
    
    % Custom 2D spatial coherence function
    spatialCoherence = calculateSpatialCoherence(center, options);
    
    % Calculate combined objective with spatial coherence
    fcmObjective = sum(sum((dist.^2) .* max(memFcnMat, eps)));
    objFcn = fcmObjective + options.lambda * spatialCoherence;
    
    % Update fuzzy partition matrix
    tmp = (max(dist, eps)).^(-2 / (expo - 1));
    newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));
end

function info = saveClusterInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k)
    info.ClusterCenters{k} = center;
    info.FuzzyPartitionMatrix{k} = fuzzyPartMat;
    info.ObjectiveFcnValue{k} = objFcn;
    info.CovarianceMatrix{k} = covMat;
    info.ValidityIndex(k) = validityIndex;
end

function lastResults = saveOptimalResults(center, fuzzyPartMat, objFcn, covMat, validityIndex, k)
    lastResults.center = center;
    lastResults.fuzzyPartMat = fuzzyPartMat;
    lastResults.objFcn = objFcn;
    lastResults.covMat = covMat;
    lastResults.validityIndex = validityIndex;
    lastResults.k = k;
end
