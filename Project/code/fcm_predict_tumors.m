% Specify the directory containing the image and mask files
dataDir = '../data/';

% Get a list of all image and mask files in the directory
imageFiles = dir(fullfile(dataDir, 'vs_gk_*_t1_3D_aligned_volume.nii'));
maskFiles = dir(fullfile(dataDir, 'vs_gk_*_t1_aligned_vol_mask.nii'));

% Initialize cell arrays to store paths for images and masks
imagePaths = {};
maskPaths = {};

% Pair image and mask files based on matching identifiers
for i = 1:numel(imageFiles)
    % Full path of the image file
    imagePath = fullfile(dataDir, imageFiles(i).name);
    
    % Extract the full identifier (e.g., 'vs_gk_36_t1') from the image file name
    identifier = regexp(imageFiles(i).name, 'vs_gk_\d+_t1', 'match', 'once');
    
    % Find the corresponding mask file with the exact identifier
    maskIdx = find(contains({maskFiles.name}, identifier));
    
    % Check if exactly one mask file is found for the identifier
    if numel(maskIdx) == 1
        % Full path of the mask file
        maskPath = fullfile(dataDir, maskFiles(maskIdx).name);
        
        % Store the matched paths
        imagePaths{end + 1} = imagePath;
        maskPaths{end + 1} = maskPath;
    elseif numel(maskIdx) > 1
        % Print a warning if multiple mask files are found
        fprintf('Warning: Multiple masks found for %s\n', identifier);
    else
        % Print a warning if no matching mask file is found
        fprintf('Warning: No mask found for %s\n', imageFiles(i).name);
    end
end

% Display the matched pairs to verify correctness
for j = 1:numel(imagePaths)
    fprintf('Image: %s\n', imagePaths{j});
    fprintf('Mask: %s\n\n', maskPaths{j});
end



% Split data into training and test sets (e.g., 80% training, 20% test)
numSamples = numel(imagePaths);
randIndices = randperm(numSamples);
numTrain = round(0.8 * numSamples);

trainIndices = randIndices(1:numTrain);
testIndices = randIndices(numTrain + 1:end);

trainingImagePaths = imagePaths(trainIndices);
trainingMaskPaths = maskPaths(trainIndices);

testImagePaths = imagePaths(testIndices);
testMaskPaths = maskPaths(testIndices);

% Display the number of samples in each set
fprintf('Number of training samples: %d\n', numel(trainingImagePaths));
fprintf('Number of test samples: %d\n', numel(testImagePaths));
%%
% Initialize variables to accumulate intensity values, spatial information, and areas
intensityValues = [];
centroids = [];
boundingBoxDiagonals = [];
tumorAreas = [];  % To store area of tumor regions for min and max area calculation

% Loop through the training set
for i = 1:numel(trainingImagePaths)
    % Load image and mask
    fprintf("Image Data:%s\n", trainingImagePaths{i});
    fprintf("Mask:%s\n", trainingMaskPaths{i});
    imageData = niftiread(trainingImagePaths{i});
    maskData = niftiread(trainingMaskPaths{i});
    
    % Convert to double for processing
    imageData = double(imageData);
    maskData = logical(maskData);  % Ensure mask is binary
    
    % Find the slice with the largest tumor region
    tumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(tumorPixelCounts);
    sliceData = imageData(:, :, maxSliceIndex);
    maskSlice = maskData(:, :, maxSliceIndex);
    
    % Collect intensity values within the tumor region
    intensityValues = [intensityValues; sliceData(maskSlice)];
    
    % Calculate the centroid, bounding box, and area of the tumor region in this slice
    stats = regionprops(maskSlice, 'Centroid', 'BoundingBox', 'Area');
    if ~isempty(stats)
        % Centroid
        centroids = [centroids; stats.Centroid];
        
        % Bounding box diagonal
        boundingBox = stats.BoundingBox;
        boundingBoxDiagonal = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
        boundingBoxDiagonals = [boundingBoxDiagonals; boundingBoxDiagonal];
        
        % Tumor area
        tumorAreas = [tumorAreas; stats.Area];
    end
end

% Calculate tumorIntensityRange as the min and max of all intensity values
tumorIntensityRange = [min(intensityValues), max(intensityValues)];

% Calculate roiCenter as the mean of the centroids
roiCenter = mean(centroids, 1);

% Calculate roiRadius as the mean of bounding box diagonals, representing a typical tumor size
roiRadius = mean(boundingBoxDiagonals) / 2;  % Approximate radius

% Calculate minArea and maxArea based on the training set tumor areas
minArea = min(tumorAreas);
maxArea = max(tumorAreas);

% Display the extracted information
fprintf('Tumor Intensity Range: [%f, %f]\n', tumorIntensityRange(1), tumorIntensityRange(2));
fprintf('ROI Center: [%.2f, %.2f]\n', roiCenter(1), roiCenter(2));
fprintf('ROI Radius: %.2f\n', roiRadius);
fprintf('Min Tumor Area: %.2f\n', minArea);
fprintf('Max Tumor Area: %.2f\n', maxArea);


%%
% Parameters derived from training set
nClusters = 5;  % Number of clusters for FCM

% Assume these values were calculated from the training set as shown previously
% tumorIntensityRange = [minIntensity, maxIntensity]; % Calculated from training
% roiCenter = [xCenter, yCenter]; % Calculated from training
% roiRadius = radius; % Calculated from training
% minArea = minimum tumor area observed in training set
% maxArea = maximum tumor area observed in training set

detectedMasks = cell(size(testImagePaths));  % Store results for each test image

for i = 1:numel(testImagePaths)
    % Load the test image and ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the original ground truth mask
    
    % Calculate maxSliceIndex based on the slice with the largest area
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    
    % Select the slice and the corresponding mask slice
    sliceData = imageData(:, :, maxSliceIndex);
    maskSlice = logical(maskData(:, :, maxSliceIndex));
    
    % Call the detection function with test image and learned parameters
    detectedMask = fcm_predict_tumor(testImagePaths{i}, nClusters, tumorIntensityRange, minArea, maxArea, roiCenter, roiRadius, maxSliceIndex);
    
    % Store the detected mask for further use if needed
    detectedMasks{i} = detectedMask;
    
    % Display the original slice with the ground truth tumor overlay
    figure;
    subplot(1, 2, 1);
    imshow(sliceData, []);
    hold on;
    % Create a red overlay for the ground truth mask
    originalOverlay = cat(3, ones(size(maskSlice)), zeros(size(maskSlice)), zeros(size(maskSlice)));
    hOriginal = imshow(originalOverlay);
    set(hOriginal, 'AlphaData', maskSlice * 0.5);  % Adjust transparency for visibility
    title('Original Test Slice with Ground Truth Tumor Overlay');
    hold off;
    
    % Display the same slice with the predicted tumor overlay
    subplot(1, 2, 2);
    imshow(sliceData, []);
    hold on;
    % Create a red overlay for the predicted tumor region
    predictedOverlay = cat(3, ones(size(detectedMask)), zeros(size(detectedMask)), zeros(size(detectedMask)));
    hPredicted = imshow(predictedOverlay);
    set(hPredicted, 'AlphaData', detectedMask * 0.5);  % Adjust transparency for visibility
    title('Original Test Slice with Predicted Tumor Overlay');
    hold off;
end


%%
% Histogram Matching
% Specify the path to the known reference image and mask
referenceImagePath = fullfile(dataDir, 'vs_gk_5_t1_3D_aligned_volume.nii');
referenceMaskPath = fullfile(dataDir, 'vs_gk_5_t1_aligned_vol_mask.nii');

% Load the reference image and mask
referenceImageData = niftiread(referenceImagePath);
referenceMaskData = niftiread(referenceMaskPath);

% Convert the reference mask to logical for processing
referenceMaskData = logical(referenceMaskData);

% Calculate maxSliceIndex for the reference image based on the largest tumor area
tumorPixelCountsRef = squeeze(sum(sum(referenceMaskData, 1), 2));
[~, maxSliceIndexRef] = max(tumorPixelCountsRef);

% Select the reference slice for histogram matching
referenceSlice = referenceImageData(:, :, maxSliceIndexRef);

% Parameters derived from training set
nClusters = 5;  % Number of clusters for FCM

% Assume these values were calculated from the training set as shown previously
% tumorIntensityRange = [minIntensity, maxIntensity]; % Calculated from training
% roiCenter = [xCenter, yCenter]; % Calculated from training
% roiRadius = radius; % Calculated from training
% minArea = minimum tumor area observed in training set
% maxArea = maximum tumor area observed in training set

detectedMasks = cell(size(testImagePaths));  % Store results for each test image

for i = 1:numel(testImagePaths)
    % Load the test image and its ground truth mask
    imageData = niftiread(testImagePaths{i});
    maskData = niftiread(testMaskPaths{i});  % Load the ground truth mask
    
    % Convert mask to logical format
    maskData = logical(maskData);

    % Calculate maxSliceIndex based on the slice with the largest area in the test mask
    sliceTumorPixelCounts = squeeze(sum(sum(maskData, 1), 2));
    [~, maxSliceIndex] = max(sliceTumorPixelCounts);
    
    % Select the test slice and match its histogram to the reference slice
    testSlice = imageData(:, :, maxSliceIndex);
    matchedSlice = imhistmatch(testSlice, referenceSlice);  % Histogram match to reference

    % Replace the slice in the image data with the histogram-matched version
    imageData(:, :, maxSliceIndex) = matchedSlice;
    
    % Call the detection function with the histogram-matched test image and learned parameters
    detectedMask = fcm_predict_tumor(testImagePaths{i}, nClusters,...
        tumorIntensityRange, minArea, maxArea, roiCenter, roiRadius, maxSliceIndex, true);
    
    % Store the detected mask for further use if needed
    detectedMasks{i} = detectedMask;
    
    % Display the original slice with the ground truth tumor overlay
    figure;
    subplot(1, 3, 1);
    imshow(testSlice, []);
    hold on;
    % Create a red overlay for the ground truth mask
    originalOverlay = cat(3, ones(size(maskData(:, :, maxSliceIndex))), zeros(size(maskData(:, :, maxSliceIndex))), zeros(size(maskData(:, :, maxSliceIndex))));
    hOriginal = imshow(originalOverlay);
    set(hOriginal, 'AlphaData', maskData(:, :, maxSliceIndex) * 0.5);  % Adjust transparency for visibility
    title('Original Test Slice with Ground Truth Tumor Overlay');
    hold off;
    
    % Display the histogram-matched test slice
    subplot(1, 3, 2);
    imshow(matchedSlice, []);
    title('Histogram Matched Test Slice');
    
    % Display the detected tumor overlay on the original test slice
    subplot(1, 3, 3);
    imshow(testSlice, []);
    hold on;
    % Create a red overlay for the predicted tumor region
    predictedOverlay = cat(3, ones(size(detectedMask)), zeros(size(detectedMask)), zeros(size(detectedMask)));
    hPredicted = imshow(predictedOverlay);
    set(hPredicted, 'AlphaData', detectedMask * 0.5);  % Adjust transparency for visibility
    title('Detected Tumor Region After FCM');
    hold off;
end


%%
function detectedTumorMask = fcm_predict_tumor(imagePath, nClusters,...
    tumorIntensityRange, minArea, maxArea, roiCenter, roiRadius, maxSliceIndex,...
    useBAT)
    % Load the image
    volumeData = niftiread(imagePath);
    sliceData = volumeData(:, :, maxSliceIndex);  % Use the slice of interest
    preprocessedSliceData = imadjust(mat2gray(sliceData));
    data = double(preprocessedSliceData(:));

    rng("default")
    % Apply FCM Clustering
    if useBAT
    % Apply FCM Clustering
    options = struct();
    options.NumClusters = nClusters;
    options.ClusterCenters = [];
    options.Exponent = 2;
    options.MaxNumIteration = 10;
    options.DistanceMetric = 'euclidean';
    options.MinImprovement = 1e-5;
    options.Verbose = 1;
    options.ClusterVolume = 1;
    options.lambda = 0.1;
    options.alpha = 1;
    options.beta = 0.5;
    options.zeta = 1.5;
    
    % BAT Options.
    options.nBats = 50;
    options.BATIterMax = 10;
    options.lowerBound = min(data);
    options.upperBound = max(data);
    options.Qmin = 0;
    options.Qmax = 2;
    options.loudness = 0.5;
    options.loudnessCoefficient = .9;
    options.pulseRate = 0.5;
    options.gamma = 0.95;
    options.chaotic = false;
    options.MinNumIteration = 50;
    options.UsePerturbation = false;
    options.PerturbationFactor = 0.01;
    
    % Apply BAT + Fuzzy C-Means (FCM) clustering
    segImgInresults = MFBAFCM(data, options);
    clusterCenters = segImgInresults.centers;
    membership = segImgInresults.U;
    else
     options = fcmOptions(NumClusters=nClusters, MaxNumIteration=10);
     [clusterCenters, membership] = fcm(double(preprocessedSliceData(:)), options);
    end 

    % Reshape membership to match image dimensions
    [~, maxMembership] = max(membership, [], 1);
    segmentedSlice = reshape(maxMembership, size(preprocessedSliceData));
    
    % Select the cluster that falls within the tumor intensity range
    tumorClusterIdx = find(clusterCenters >= tumorIntensityRange(1) & clusterCenters <= tumorIntensityRange(2));
    if isempty(tumorClusterIdx)
        warning('No cluster found within the tumor intensity range.');
        detectedTumorMask = false(size(preprocessedSliceData));
        return;
    else
        % Create an initial mask for the detected tumor region
        detectedTumorMask = ismember(segmentedSlice, tumorClusterIdx);
    end
    
    % Apply spatial constraints: Restrict to ROI based on training statistics
    [X, Y] = meshgrid(1:size(preprocessedSliceData, 2), 1:size(preprocessedSliceData, 1));
    distanceFromCenter = sqrt((X - roiCenter(1)).^2 + (Y - roiCenter(2)).^2);
    spatialMask = distanceFromCenter <= roiRadius;
    detectedTumorMask = detectedTumorMask & spatialMask;

    % Filter based on size constraints derived from training data
    detectedTumorMask = bwareafilt(detectedTumorMask, [minArea, maxArea]);

    % Post-processing (e.g., adaptive dilation based on training size statistics)
    tumorStats = regionprops(detectedTumorMask, 'BoundingBox');
    if ~isempty(tumorStats)
        boundingBox = tumorStats.BoundingBox;
        boundingBoxDiameter = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
        dilationRadius = round(boundingBoxDiameter * 0.1);  % Adjust based on training data
        detectedTumorMask = imdilate(detectedTumorMask, strel('disk', dilationRadius));
    end
end
%%
function varargout = customFCM(data, options)
    % Fuzzy c-means clustering with optional automatic cluster determination.

    dataSize = size(data, 1);
    maxAutoK = 10;
    objFcn = zeros(options.MaxNumIteration, 1);
    kValues = determineClusterRange(options, dataSize, maxAutoK);
    
    % Initialize info structure for storing results
    info = initializeInfoStruct(numel(kValues));
    centers = options.ClusterCenters;
    minKIndex = Inf;
    lastResults = [];

    % Perform clustering for each k in kValues
    for ct = 1:numel(kValues)
        k = kValues(ct);
        options = adjustOptions(options, k, centers, data);
        fuzzyPartMat = fuzzy.clustering.initfcm(options, dataSize);

        % Iterate to find cluster centers and partition matrix
        [center, objFcn, fuzzyPartMat, covMat] = iterateClustering(data, options, fuzzyPartMat, objFcn);

        % Calculate validity index and update best result if optimal
        validityIndex = fuzzy.clustering.clusterValidityIndex(data, center, fuzzyPartMat);
        info = updateInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, ct);
        [minKIndex, lastResults] = updateBestResults(lastResults, center,...
            fuzzyPartMat, objFcn, covMat, validityIndex, k, minKIndex, ct);

        % Update centers for next iteration
        if ~isempty(options.ClusterCenters)
            centers = lastResults.center;
        end
    end

    % Finalize info structure and remove unnecessary fields if Euclidean
    info.OptimalNumClusters = lastResults.k;
    if strcmp(options.DistanceMetric, 'euclidean')
        info = rmfield(info, "CovarianceMatrix");
    end

    % Assign outputs
    [varargout{1:nargout}] = assignOutputs(info, minKIndex);
end

% Helper Functions

function kValues = determineClusterRange(options, dataSize, maxAutoK)
    % Determine range of cluster numbers based on data size and options
    if isequal(options.NumClusters, fuzzy.clustering.FCMOptions.DefaultNumClusters)
        kStart = max(2, size(options.ClusterCenters, 1));
        maxNumClusters = min(kStart + maxAutoK - 1, dataSize - 1);
        kValues = kStart:maxNumClusters;
    else
        kValues = options.NumClusters;
    end
end

function info = initializeInfoStruct(numFolds)
    % Initialize info structure to store results for each k
    info = struct(...
        'NumClusters', zeros(1, numFolds), ...
        'ClusterCenters', cell(1, numFolds), ...
        'FuzzyPartitionMatrix', cell(1, numFolds), ...
        'ObjectiveFcnValue', cell(1, numFolds), ...
        'CovarianceMatrix', cell(1, numFolds), ...
        'ValidityIndex', zeros(1, numFolds), ...
        'OptimalNumClusters', 0);
end

function options = adjustOptions(options, k, centers, data)
    % Adjust options for current cluster count k
    options.ClusterCenters = [];
    options.NumClusters = k;
    options.ClusterVolume = adjustClusterVolume(options.ClusterVolume, k);
    options.ClusterCenters = initializeCenters(centers, k, data);
end

function clusterVolume = adjustClusterVolume(clusterVolume, k)
    % Adjust the cluster volume for each cluster if necessary
    if numel(clusterVolume) ~= k
        clusterVolume = repmat(clusterVolume(1), 1, k);
    end
end

function centers = initializeCenters(centers, k, data)
    % Initialize cluster centers if needed
    if ~isempty(centers) && size(centers, 1) < k
        fprintf("Init. centers using kmeans\n")
        centers = fuzzy.clustering.addKMPPCenters(data, centers, k);
    end
end

function [center, objFcn, fuzzyPartMat, covMat] = iterateClustering(data, options, fuzzyPartMat, objFcn)
    % Perform clustering iterations to optimize centers and partition matrix
    iterationProgressFormat = getString(message('fuzzy:general:lblFcm_iterationProgressFormat'));
    for iterId = 1:options.MaxNumIteration
        [center, fuzzyPartMat, objFcn(iterId), covMat, stepBrkCond, options] = ...
            stepfcm(data,fuzzyPartMat,options);
        brkCond = checkBreakCondition(options,objFcn(iterId:-1:max(1,iterId-1)),iterId,stepBrkCond);
        % Check verbose condition
        if options.Verbose
            fprintf(iterationProgressFormat, iterId, objFcn(iterId));
            if ~isempty(brkCond.description)
                fprintf('%s\n',brkCond.description);
            end
        end

        % Break if early termination condition is true.
        if brkCond.isTrue
            objFcn(iterId+1:end) = [];
            break
        end
    end
end

function info = updateInfo(info, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, ct)
    % Store clustering results in info structure
    info.NumClusters(ct) = k;
    info.ClusterCenters{ct} = center;
    info.FuzzyPartitionMatrix{ct} = fuzzyPartMat;
    info.ObjectiveFcnValue{ct} = objFcn;
    info.CovarianceMatrix{ct} = covMat;
    info.ValidityIndex(ct) = validityIndex;
end

function [minKIndex, lastResults] = updateBestResults(lastResults, center, fuzzyPartMat, objFcn, covMat, validityIndex, k, minKIndex, ct)
    % Update best clustering results based on validity index
    if isempty(lastResults) || validityIndex < lastResults.validityIndex
        lastResults = struct('center', center, 'fuzzyPartMat', fuzzyPartMat, 'objFcn', objFcn, ...
                             'covMat', covMat, 'validityIndex', validityIndex, 'k', k);
        minKIndex = ct;
    end
end

function varargout = assignOutputs(info, minIndex)
    % Assign function outputs
    if nargout > 3, varargout{4} = info; end
    if nargout > 2, varargout{3} = info.ObjectiveFcnValue{minIndex}; end
    if nargout > 1, varargout{2} = info.FuzzyPartitionMatrix{minIndex}; end
    if nargout > 0, varargout{1} = info.ClusterCenters{minIndex}; end
end

function brkCond = checkBreakCondition(options,objFcn,iterId,stepBrkCond)

if stepBrkCond.isTrue
    brkCond = stepBrkCond;
    return
end

brkCond = struct('isTrue',false,'description','');
improvement = diff(objFcn);
if ~isempty(improvement) && (abs(improvement)<=options.MinImprovement || isnan(improvement))
    brkCond.isTrue = true;
    brkCond.description = getString(message('fuzzy:general:msgFcm_minImprovementReached'));
    return
end
if iterId==options.MaxNumIteration
    brkCond.isTrue = true;
    brkCond.description = getString(message('fuzzy:general:msgFcm_maxIterationReached'));
end
end

function [center, newFuzzyPartMat, objFcn, covMat, brkCond,...
    options] = stepfcm(data, fuzzyPartMat, options)
    % One step in fuzzy c-means clustering with a custom objective function.

    % Extract parameters from options
    numCluster = options.NumClusters;
    expo = options.Exponent;
    lambda = options.lambda;
    clusterVolume = options.ClusterVolume;
    brkCond = struct('isTrue', false, 'description', '');

    % Update the fuzzy partition matrix with the exponent
    memFcnMat = fuzzyPartMat .^ expo;

    % Compute or adjust cluster centers
    if isempty(options.ClusterCenters)
        center = (memFcnMat * data) ./ (sum(memFcnMat, 2) * ones(1, size(data, 2)));
        fprintf("Update centers\n");
    else
        fprintf("Use given centers\n");
        center = options.ClusterCenters;
        if options.UsePerturbation
             fprintf("Perturb\n");
            center = center + options.PerturbationFactor * randn(size(center));
        end
        options.ClusterCenters = [];
    end

    % Calculate distances and covariance matrix based on the selected metric
    switch options.DistanceMetric
        case 'mahalanobis'
            [dist, covMat, brkCond] = fuzzy.clustering.mahalanobisdist(center, data, memFcnMat, clusterVolume);
        case 'fmle'
            [dist, covMat, brkCond] = fuzzy.clustering.fmle(center, data, memFcnMat);
        otherwise
            dist = fuzzy.clustering.euclideandist(center, data);
            covMat = [];
    end

    % Calculate the traditional FCM objective function
    %fcmObjective = sum(sum((dist.^2) .* max(memFcnMat, eps)));

    % Calculate the custom fitness value
    %fitnessValue = calculateFitness(center, data, options);

    % Combine the traditional FCM objective and fitness function
    %fprintf("Lambda:%5.2f\n", lambda);
    %objFcn = fcmObjective + lambda * fitnessValue;

    % Calculate custom objective function
    objFcn = calculateFitness(center, data, options);

    % Update the fuzzy partition matrix
    tmp = max(dist, eps) .^ (-2 / (expo - 1));
    newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));
end



%%
function fitness = calculateFitness(clusterCenters, data, options)
    % Optimized fitness calculation based on intra-cluster distance, SC, PC, and CE

    m = options.Exponent;  % Fuzziness exponent
    alpha = options.alpha;
    beta = options.beta;
    zeta = options.zeta;

    % Compute squared distances between data points and cluster centers
    distances = max(pdist2(data, clusterCenters, 'squaredeuclidean'), 1e-10);

    % Calculate membership matrix U
    U = calculateMembership(distances, m);

    % Compute fitness components
    intraCluster = calculateIntraCluster(data, clusterCenters, U, m);
    SC = calculatePartitionIndex(U, distances, clusterCenters, m);
    PC = calculatePartitionCoefficient(U);
    CE = calculateClassificationEntropy(U);

    % Final fitness value
    fitness = alpha * intraCluster + beta * SC + zeta * (1 / PC + CE);
end

function U = calculateMembership(distances, m)
    % Calculate membership matrix U with fuzziness exponent m
    exponent = 2 / (m - 1);
    invDistances = 1 ./ distances;
    U = (invDistances .^ exponent) ./ sum(invDistances .^ exponent, 2);
end

function intraCluster = calculateIntraCluster(dataPoints, clusterCenters, U, m)
    % Vectorized calculation of weighted intra-cluster distance
    % U: N x c membership matrix, raised to the power m
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');
    intraCluster = sum(sum((U .^ m) .* distances)) / size(dataPoints, 1);
end

function SC = calculatePartitionIndex(U, distances, clusterCenters, m)
    % Calculate the Partition Index (SC) based on intra- and inter-cluster distances
    % intra-cluster part
    intraClusterDist = sum((U .^ m) .* distances, 1);  % 1 x c vector

    % inter-cluster part
    clusterDistances = max(pdist2(clusterCenters, clusterCenters, 'squaredeuclidean'), 1e-10);
    N = size(U, 1);  % Number of data points
    denominator = N * sum(clusterDistances, 2)';  % 1 x c vector

    SC = sum(intraClusterDist ./ denominator);  % Sum for all clusters
end

function PC = calculatePartitionCoefficient(U)
    % Calculate Partition Coefficient (PC)
    N = size(U, 1);
    PC = sum(U .^ 2, 'all') / N;
end

function CE = calculateClassificationEntropy(U)
    % Calculate Classification Entropy (CE)
    epsilon = 1e-10;  % Small value to avoid log(0)
    N = size(U, 1);
    CE = -sum(U .* log(U + epsilon), 'all') / N;
end

%%
function [bestClusterCenters, bestFitness] = batAlgorithm(data, options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rates
    loudnesses = options.loudness * ones(options.nBats, 1);   % Per-bat loudness


    numFeatures = size(data, 2);
    % Initialize bat positions with bounds expanded to (nBats x (numClusters * numFeatures))
    bats = repmat(options.lowerBound, options.nBats, options.NumClusters) + ...
       (repmat(options.upperBound, options.nBats, options.NumClusters) - ...
       repmat(options.lowerBound, options.nBats, options.NumClusters)) ...
       .* rand(options.nBats, options.NumClusters * numFeatures);

    velocities = zeros(options.nBats, options.NumClusters * numFeatures);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats using the dataset
    fprintf("BAT Initialization\n");
    for i = 1:options.nBats
         fprintf("BAT:%d\n", i);
        % Reshape each bat into a (numClusters x calculateFitness) format for calculateFitness
        reshapedBat = reshape(bats(i, :), [options.NumClusters, numFeatures]);
        fitness(i) = calculateFitness(reshapedBat, data, options);  % Calculate fitness based on clustering
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestClusterCenters = bats(idx, :);  % Store best solution as flattened vector

    % Main loop for the BAT algorithm
    for t = 1:options.BATIterMax
        fprintf("BAT Iter:%d\n", t);
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            Q = options.Qmin + (options.Qmax - options.Qmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestClusterCenters) * Q;
            newClusterCenters = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints for each cluster center
            newClusterCenters = enforceBoundaries(newClusterCenters,...
                options.lowerBound, options.upperBound,...
                options.NumClusters, numFeatures);

            % Local search (small random walk around the best solution)
            if rand > pulseRates(i)
                newClusterCenters = bestClusterCenters + ...
                    0.01 * randn(1, options.NumClusters * numFeatures);
            end

            % Evaluate the new solution's fitness with the dataset
            reshapedNewClusterCenters = reshape(newClusterCenters, [options.NumClusters, numFeatures]);
            newFitness = calculateFitness(reshapedNewClusterCenters, data, options);

            % Acceptance criteria based on fitness, loudness, and pulse rate
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                fprintf("Update bats, fitness, loudness, pulse rates\n")
                bats(i, :) = newClusterCenters;
                fitness(i) = newFitness;
                loudnesses(i) = options.loudnessCoefficient * loudnesses(i);  % Decrease loudness
                pulseRates(i) = pulseRates(i) * (1 - exp(-options.gamma * t));  % Increase pulse rate
            end

            % Update global best if a better solution is found
            if newFitness < bestFitness
                fprintf("Update best centers\n")
                bestClusterCenters = newClusterCenters;
                bestFitness = newFitness;
            end
        end
    end

    % Reshape bestSol into (numClusters x 9) for output
    bestClusterCenters = reshape(bestClusterCenters, [options.NumClusters, numFeatures]);
end

% Function to enforce boundary constraints for each cluster center
function clusterCenters = enforceBoundaries(clusterCenters,...
    lowerBound, upperBound, numClusters, numFeatures)
    % Reshape to (numClusters x numFeatures) for easy boundary enforcement
    reshapedSolution = reshape(clusterCenters, [numClusters, numFeatures]);
    % Apply boundary constraints for each dimension
    boundedSolution = max(min(reshapedSolution, upperBound), lowerBound);
    % Flatten back to 1D array
    clusterCenters = boundedSolution(:)';
end

function results = MFBAFCM(data, options)
    % Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [batCenters, ~] = batAlgorithm(data, options);

    options.ClusterCenters = batCenters;
    disp('FCM Options')
    disp(options)
    [centers,U,objFcn,info] = customFCM(data, options);
    results = struct();
    results.U = U;
    results.batCenters = batCenters;
    results.centers = centers;
    results.objFcn = objFcn;
    results.info = info;
end