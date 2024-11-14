clearvars 
clearvars 
clc

% Specify the path to your NIfTI file (3D volume)
niftiFilePath = '../data/vs_gk_36_t1_3D_aligned_volume.nii';
%niftiFilePath = '../data/vs_gk_5_t1_3D_aligned_volume.nii';

% Load the NIfTI image data (3D volume)
volumeData = niftiread(niftiFilePath);
volumeData = double(volumeData);

% Specify the path to the NIfTI file for the binary mask
maskFilePath = '../data/vs_gk_36_t1_aligned_vol_mask.nii';
%maskFilePath = '../data/vs_gk_5_t1_aligned_vol_mask.nii';

% Load binary mask
binaryMask = niftiread(maskFilePath);
binaryMask = double(binaryMask);  % Convert to double for processing

% Find the slice with the largest tumor region
tumorPixelCounts = squeeze(sum(sum(binaryMask, 1), 2));  % Sum tumor pixels in each slice
[maxPixels, maxSliceIndex] = max(tumorPixelCounts);  % Find the slice with the most tumor pixels

if maxPixels == 0
    warning('No tumor pixels found in the entire volume.');
else
    % Display the selected slice with the largest tumor area
    figure;
    subplot(1, 2, 1);
    imshow(volumeData(:, :, maxSliceIndex), []);
    title(['Original Slice ', num2str(maxSliceIndex)]);

    % Overlay the tumor on the original image
    subplot(1, 2, 2);
    imshow(volumeData(:, :, maxSliceIndex), []);
    hold on;

    % Create a red overlay for the tumor
    tumorOverlay = binaryMask(:, :, maxSliceIndex);
    redOverlay = cat(3, ones(size(tumorOverlay)), zeros(size(tumorOverlay)), zeros(size(tumorOverlay)));  % Red overlay
    h = imshow(redOverlay);
    set(h, 'AlphaData', tumorOverlay * 0.5);  % Set transparency for visibility

    title(['Tumor Overlay on Slice ', num2str(maxSliceIndex)]);
    hold off;
end


%%
% Apply Fuzzy C-Means clustering on the selected slice
    % Load and Preprocess Slice
sliceData = volumeData(:, :, maxSliceIndex);
preprocessedSliceData = imadjust(mat2gray(sliceData));

% Apply FCM Clustering
nClusters = 5;
options = fcmOptions(NumClusters=nClusters, MaxNumIteration=100);
[clusterCenters, membership] = fcm(double(preprocessedSliceData(:)), options);

% Ensure that tumorOverlay is a logical mask
tumorOverlay = logical(binaryMask(:, :, maxSliceIndex));

% Label connected components in tumorOverlay
labeledComponents = bwlabel(tumorOverlay);

% Find the most common label in the tumor region to get the largest component
tumorLabel = mode(labeledComponents(tumorOverlay & labeledComponents > 0));

% Define tumorComponent as the largest connected component
tumorComponent = (labeledComponents == tumorLabel);

% Calculate median intensity of the tumor component in preprocessedSliceData
tumorIntensityMedian = median(preprocessedSliceData(tumorComponent));

% Find cluster center closest to tumor intensity median
[~, tumorCluster] = min(abs(clusterCenters - tumorIntensityMedian));

% Reshape the membership values to match slice dimensions
[~, maxMembership] = max(membership, [], 1);
segmentedSlice = reshape(maxMembership, size(preprocessedSliceData));

% Generate binary mask for detected tumor region
detectedTumorMask = (segmentedSlice == tumorCluster);

% Limit detection to regions near the tumor component
dilatedTumorComponent = imdilate(tumorComponent, strel('disk', 10));  % Expand area slightly for spatial proximity
detectedTumorMask = detectedTumorMask & dilatedTumorComponent;

% Post-process to retain only the largest connected component
detectedTumorMask = bwareafilt(detectedTumorMask, 1);

% Calculate the bounding box of the detected tumor mask
tumorStats = regionprops(detectedTumorMask, 'BoundingBox');

% If tumorStats is empty, handle the case where no tumor is detected
if ~isempty(tumorStats)
    % Use the diagonal of the bounding box to determine dilation radius
    boundingBox = tumorStats.BoundingBox;
    boundingBoxDiameter = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
    dilationRadius = round(boundingBoxDiameter * 0.1);  % Adjust 0.1 as a scaling factor
    
    % Apply dilation with the calculated radius
    expandedTumorMask = imdilate(detectedTumorMask, strel('disk', dilationRadius));
else
    warning('No tumor detected in the mask.');
    expandedTumorMask = detectedTumorMask;  % No expansion if no tumor detected
end



%%
% Display original and detected masks
figure;
subplot(1, 2, 1);
imshow(tumorOverlay, []);
title('Original Tumor Mask');

subplot(1, 2, 2);
imshow(preprocessedSliceData, []);
hold on;
redOverlay = cat(3, ones(size(expandedTumorMask)),...
    zeros(size(expandedTumorMask)), zeros(size(detectedTumorMask)));
h = imshow(redOverlay);
set(h, 'AlphaData', expandedTumorMask * 0.5);
title('Detected Tumor Region After FCM');
hold off;
%%
% Apply Fuzzy C-Means clustering on the selected slice
% Load and Preprocess Slice
sliceData = volumeData(:, :, maxSliceIndex);
preprocessedSliceData = imadjust(mat2gray(sliceData));
data = double(preprocessedSliceData(:));

% Apply FCM Clustering
rng("default")
nClusters = 5;  % Adjust this parameter based on the data
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

% nClusters = 5;
% options = fcmOptions(NumClusters=nClusters, MaxNumIteration=100);
% [clusterCenters, membership] = fcm(double(preprocessedSliceData(:)), options);

% Ensure that tumorOverlay is a logical mask
tumorOverlay = logical(binaryMask(:, :, maxSliceIndex));

% Label connected components in tumorOverlay
labeledComponents = bwlabel(tumorOverlay);

% Find the most common label in the tumor region to get the largest component
tumorLabel = mode(labeledComponents(tumorOverlay & labeledComponents > 0));

% Define tumorComponent as the largest connected component
tumorComponent = (labeledComponents == tumorLabel);

% Calculate median intensity of the tumor component in preprocessedSliceData
tumorIntensityMedian = median(preprocessedSliceData(tumorComponent));

% Find cluster center closest to tumor intensity median
[~, tumorCluster] = min(abs(segImgInresults.centers - tumorIntensityMedian));

% Reshape the membership values to match slice dimensions
[~, maxMembership] = max(segImgInresults.U, [], 1);
segmentedSlice = reshape(maxMembership, size(preprocessedSliceData));

% Generate binary mask for detected tumor region
detectedTumorMask = (segmentedSlice == tumorCluster);

% Limit detection to regions near the tumor component
dilatedTumorComponent = imdilate(tumorComponent, strel('disk', 10));  % Expand area slightly for spatial proximity
detectedTumorMask = detectedTumorMask & dilatedTumorComponent;

% Post-process to retain only the largest connected component
detectedTumorMask = bwareafilt(detectedTumorMask, 1);

% Calculate the bounding box of the detected tumor mask
tumorStats = regionprops(detectedTumorMask, 'BoundingBox');

% If tumorStats is empty, handle the case where no tumor is detected
if ~isempty(tumorStats)
    % Use the diagonal of the bounding box to determine dilation radius
    boundingBox = tumorStats.BoundingBox;
    boundingBoxDiameter = sqrt(boundingBox(3)^2 + boundingBox(4)^2);
    dilationRadius = round(boundingBoxDiameter * 0.1);  % Adjust 0.1 as a scaling factor
    
    % Apply dilation with the calculated radius
    expandedTumorMask = imdilate(detectedTumorMask, strel('disk', dilationRadius));
else
    warning('No tumor detected in the mask.');
    expandedTumorMask = detectedTumorMask;  % No expansion if no tumor detected
end

%%
% Display original and detected masks
figure;
subplot(1, 2, 1);
imshow(tumorOverlay, []);
title('Original Tumor Mask');

subplot(1, 2, 2);
imshow(preprocessedSliceData, []);
hold on;
redOverlay = cat(3, ones(size(expandedTumorMask)),...
    zeros(size(expandedTumorMask)), zeros(size(detectedTumorMask)));
h = imshow(redOverlay);
set(h, 'AlphaData', expandedTumorMask * 0.5);
title('Detected Tumor Region After BAT+FCM');
hold off;


%%
% Use the selected slice data for clustering
sliceData = volumeData(:, :, maxSliceIndex);
[r, c] = size(sliceData);  % Get dimensions
trainingData = reshape(sliceData, [r * c, 1]);  % Reshape to a 2D array for clustering
kDim = [3 3];
trainFeatures = createMovingWindowFeatures(trainingData, kDim);  % Extract moving window features

%% BAT + FCM Clustering Setup
rng("default")
nClusters = 15;  % Adjust this parameter based on the data
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
options.lowerBound = min(trainFeatures);
options.upperBound = max(trainFeatures);
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
segImgInresults = MFBAFCM(trainFeatures, options);

%% Post-processing: Identify Tumor Cluster with Highest Overlap
ecDist = findDistance(segImgInresults.centers, trainFeatures);
[~, ecBATFcmLabel] = min(ecDist', [], 2); 
ecBATFcmLabel = reshape(ecBATFcmLabel, [r, c]);  % Reshape back to 2D

% Calculate overlap and compactness metrics for each cluster
overlapCounts = zeros(1, nClusters);
overlapPercentage = zeros(1, nClusters);
compactnessScore = zeros(1, nClusters);

for clusterId = 1:nClusters
    clusterMask = (ecBATFcmLabel == clusterId);
    overlapCounts(clusterId) = sum(clusterMask(:) & tumorOverlay(:));  % Count overlap with tumor mask
    overlapPercentage(clusterId) = overlapCounts(clusterId) / sum(clusterMask(:)) * 100;
    
    % Calculate compactness as inverse of average distance to the cluster center
    if sum(clusterMask(:)) > 0
        clusterPoints = trainFeatures(clusterMask(:), :);
        averageDistToCenter = mean(pdist2(clusterPoints, segImgInresults.centers(clusterId, :)));
        compactnessScore(clusterId) = 1 / averageDistToCenter;  % Higher score for more compact clusters
    end
end

% Select the tumor cluster with highest overlap * compactness score
[~, tumorCluster] = max(overlapPercentage .* compactnessScore);

% Display the detected tumor region
detectedTumorRegion = (ecBATFcmLabel == tumorCluster);

figure;
subplot(1, 2, 1);
imshow(volumeData(:, :, maxSliceIndex), []);
title(['Original Slice ', num2str(maxSliceIndex)]);

subplot(1, 2, 2);
imshow(volumeData(:, :, maxSliceIndex), []);
hold on;

% Create a red overlay for the detected tumor region
redOverlayDetected = cat(3, ones(size(detectedTumorRegion)),...
    zeros(size(detectedTumorRegion)), zeros(size(detectedTumorRegion)));
h = imshow(redOverlayDetected);
set(h, 'AlphaData', detectedTumorRegion * 0.5);  % Make the detected tumor area semi-transparent

title(['Detected Tumor Region on Slice ', num2str(maxSliceIndex)]);
hold off;
%%
% Visualization of the BAT + FCM segmented results
figure;
subplot(1, 2, 1);
imshow(tumorOverlay, []);
xlabel("Original Tumor Mask");

subplot(1, 2, 2);
BATFCMImg = ecBATFcmLabel / nClusters;
imshow(BATFCMImg, []);
xlabel("BAT+FCM Segmented Tumor Region");

%% Tumor Segmentation Evaluation
refTumor = find(tumorOverlay == 1);  % Reference tumor pixel indices
tumorCluster = 3;  % Set the cluster ID you believe represents the tumor

[ecHasTumor, ecNumFalsePos, ecTumor] = segmentTumor(ecBATFcmLabel, refTumor, tumorCluster);

% Display evaluation results
fprintf('Tumor Detection Status: %d\n', ecHasTumor);
fprintf('Number of False Positives: %d\n', ecNumFalsePos);

%% Supporting Function: segmentTumor
function [hasTumor, numFalsePos, tumorLabel] = segmentTumor(testLabel, refPositiveIds, clusterId)
    % Calculate detection results using the test and reference data.

    tumorIds = testLabel == clusterId;
    segmentedImage = testLabel;
    segmentedImage(tumorIds) = 1;
    segmentedImage(~tumorIds) = 0;
    
    tumorIdsECIds = find(tumorIds == 1);
    hasTumor = ~isempty(intersect(tumorIdsECIds, refPositiveIds));  % Check if any overlap
    numFalsePos = length(setdiff(tumorIdsECIds, refPositiveIds));  % Calculate false positives
    tumorLabel = segmentedImage;
end

%%
function filteredI = denoiseImage(I)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    fprintf('Slice:%d\n', i)
    slice = double(I(:, :, i));
    %slice = imdiffusefilt(slice);  % Apply  filter
    slice =  medfilt2(slice, [3 3]);
    % Normalize between 0 and 1
    slice = (slice - min(slice(:))) / (max(slice(:)) - min(slice(:)));
    filteredI(:, :, i) = slice;  % Store the filtered slice back
 end
end

function y = createMovingWindowFeatures(in,dim)
% Create feature vectors using a moving window.

rStep = floor(dim(1)/2);
cStep = floor(dim(2)/2);

x1 = [zeros(size(in,1),rStep) in zeros(size(in,1),rStep)];
x = [zeros(cStep,size(x1,2));x1;zeros(cStep,size(x1,2))];

[row,col] = size(x);
yCol = prod(dim);
y = zeros((row-2*rStep)*(col-2*cStep), yCol);
ct = 0;
for rId = rStep+1:row-rStep
    for cId = cStep+1:col-cStep
        ct = ct + 1;
        y(ct,:) = reshape(x(rId-rStep:rId+rStep,cId-cStep:cId+cStep),1,[]);
    end
end
end


%% Calculate feature distance from cluster center.

function dist = findDistance(centers,data)

dist = zeros(size(centers, 1), size(data, 1));
for k = 1:size(centers, 1)
    dist(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*centers(k, :)).^2), 2));
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