% Load an example MRI image
% clearvars 
% clc
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Select the 15th slice of the MRI
slice_number = 15;
sliceData = D(:,:,slice_number);  % uint8 data
%sliceData = imdiffusefilt(sliceData);
%sliceData = histeq(sliceData);

% Convert uint8 data to double and normalize to [0, 1]
data = double(sliceData(:)) / 255;  % Normalizing the pixel values

nClusters = 4;

options = struct();
options.NumClusters = nClusters;
options.ClusterCenters = [];
options.Exponent = 2;
options.MaxNumIteration = 300;
options.DistanceMetric = 'euclidean';
options.MinImprovement = 1e-6;
options.Verbose = 1;
options.ClusterVolume = 1;
options.Exponent = 2;
options.dataPoints = data;
options.lambda = 0;

% BAT Options.
options.nBats = 50;
options.BATIterMax = 100;
options.lowerBound = min(data);
options.upperBound = max(data);
options.Qmin = 0;
options.Qmax = 2;
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.gamma = 0.95; % Decay rate for pulse rate
options.chaotic = false;
options.dataPoints = data;
options.MinNumIteration = 50;
options.UsePerturbation = true;
options.PerturbationFactor = 0.01;

% fcmOpt = fcmOptions(NumClusters = nClusters);
disp('FCM Options')
disp(options)

% Apply Fuzzy C-Means (FCM) clustering
% fcm(data, fcmOpt);

[fcmCenters, fcmU] = customFCM(data, options);
% Get the cluster membership for each pixel
[~, maxU] = max(fcmU);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segFCMImg = reshape(maxU, size(sliceData));


% Apply BAT + Fuzzy C-Means (FCM) clustering
segImgInresults = MFBAFCM(options);

% Get the cluster membership for each pixel
[~, maxU] = max(segImgInresults.U);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segBATFCMImg = reshape(maxU, size(sliceData));

%%
fcmUT = fcmU';
PC = calculatePartitionCoefficient(fcmUT);
CE = calculateClassificationEntropy(fcmUT);
SC = calculatePartitionIndex(fcmUT, data, ...
    fcmCenters, options.Exponent);
S = fuzzySeparationIndex(data, fcmCenters,...
    fcmU, options.Exponent);
fprintf("FCM: PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f\n", PC,CE,SC, S);
batFCMUT = segImgInresults.U';
PC = calculatePartitionCoefficient(batFCMUT);
CE = calculateClassificationEntropy(batFCMUT);
SC = calculatePartitionIndex(batFCMUT, data, ...
    segImgInresults.centers, options.Exponent);
S = fuzzySeparationIndex(data, segImgInresults.centers,...
    segImgInresults.U, options.Exponent);
fprintf("BAT+FCM PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f\n", PC,CE,SC, S);

%%
% ----- Fixed Colors for the Clusters -----
fixedColors = [
    1 0 0;   % Red for Cluster 1
    0 1 0;   % Green for Cluster 2
    0 0 1;   % Blue for Cluster 3
    1 1 0;   % Yellow for Cluster 4
];

% Display the original and clustered MRI slice
figure;
h1= subplot(1, 3, 1);
imshow(sliceData, []);  % Display the original MRI slice
title('Original MRI Slice', 'FontSize', 20, 'FontWeight','bold');

h2 = subplot(1, 3, 2);
optPlot = struct();
optPlot.title = ['Segmented MRI Slice,FCM,',num2str(nClusters), ' Clusters' ];
centers = [fcmCenters'];
optPlot.centerColors = ['kx'];
optPlot.fixedColors = fixedColors; 
optPlot.centerNames = ['FCM Centers'];
showSegmentedImg(sliceData, segFCMImg, centers, optPlot);


h3 = subplot(1, 3, 3);
optPlot = struct();
optPlot.title = ['Segmented MRI Slice,BAT + FCM', num2str(nClusters), ' Clusters'];
centers = [segImgInresults.batCenters];
optPlot.centerColors = ['rx'];
optPlot.fixedColors = fixedColors; 
optPlot.centerNames = ['Bat Centers'];
showSegmentedImg(sliceData, segBATFCMImg, centers, optPlot);

% 'Position' is [left, bottom, width, height]
set(h1, 'Position', [0.05, 0.1, 0.25, 0.8]);  % Adjust as needed
set(h2, 'Position', [0.35, 0.1, 0.25, 0.8]);  % Adjust as needed
set(h3, 'Position', [0.65, 0.1, 0.25, 0.8]);  % Adjust as needed
%%

function results = MFBAFCM(options)
    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [batCenters, ~] = batAlgorithm(options);

    options.ClusterCenters = batCenters';
    % Step 3: Run FCM with the initial cluster centers found by MBA
    %opt = fcmOptions(NumClusters = options.nClusters);
   % opt = fcmOptions(NumClusters = options.nClusters,...
   %  ClusterCenters=batCenters');
    disp('FCM Options')
    disp(options)
%    NumClusters = 'auto',...
    [centers,U,objFcn,info] = customFCM(options.dataPoints, options);
    %[centers,U,objFcn,info] = fcm(options.dataPoints, opt);

     results = struct();
     results.U = U;
     results.batCenters = batCenters;
     results.centers = centers;
     results.objFcn = objFcn;
     results.info = info;
end
%%
function [bestSol, bestFitness] = batAlgorithm(options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rates
    loudnesses = options.loudness * ones(options.nBats, 1);  % Per-bat loudness

    % Initialize bat positions for cluster centers and velocities
    bats = options.lowerBound + (options.upperBound - options.lowerBound) ...
        .* rand(options.nBats, options.NumClusters);
    velocities = zeros(options.nBats, options.NumClusters);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats
    for i = 1:options.nBats
        fitness(i) = calculateFitness(bats(i, :)', options);  % Call to calculateFitness
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestSol = bats(idx, :);

    % Main loop for the BAT algorithm
    for t = 1:options.BATIterMax
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            Q = options.Qmin + (options.Qmax - options.Qmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestSol) * Q;
            newSolution = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints
            newSolution = enforceBoundaries(newSolution, options.lowerBound, options.upperBound);

            % Local search
            if rand > pulseRates(i)
                newSolution = bestSol + 0.01 * randn(1, options.NumClusters);
            end

            % Evaluate the new solution's fitness
            newFitness = calculateFitness(newSolution', options);

            % Acceptance criteria
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                bats(i, :) = newSolution;
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

% Function to enforce boundary constraints
function newSolution = enforceBoundaries(newSolution, lowerBound, upperBound)
    newSolution = max(min(newSolution, upperBound), lowerBound);
end
%%
function fitness = calculateFitness(clusterCenters, options)
    % dataPoints: N×D (number of data points × number of features).
    % clusterCenters: C×D (number of clusters × number of features).

    % Extract the data points and number of clusters from options
    dataPoints = options.dataPoints;
    m = options.Exponent; % Fuzziness exponent

    % Calculate intra-cluster distance
    intra_cluster = calculateIntraCluster(dataPoints, clusterCenters); % No transpose needed

    % Compute distances between data points and cluster centers (N x C)
    distances = pdist2(dataPoints, clusterCenters).^2; % Squared distances

    % Avoid division by zero by setting very small values to a small epsilon
    epsilon = 1e-10;
    distances(distances < epsilon) = epsilon;

    % Update membership matrix U using vectorized operations
    exponent = 2 / (m - 1);
    invDistances = 1 ./ distances; % Element-wise inversion of distances
    sumInvDistances = sum(invDistances .^ exponent, 2); % Sum across clusters for each data point

    U = (invDistances .^ exponent) ./ sumInvDistances; % Calculate membership values

    % Calculate partition index (SC)
    SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m);

    % Calculate partition coefficient (PC)
    PC = calculatePartitionCoefficient(U);

    % Compute the fitness value
    fitness = (intra_cluster + SC) / PC;
  
end

%%
function intraCluster = calculateIntraCluster(dataPoints, clusterCenters)
    % dataPoints: N x D matrix (N data points, D features)
    % clusterCenters: c x D matrix (c cluster centers, D features)
    
    % Compute the pairwise squared Euclidean distances between data points and cluster centers
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');
    
    % Sum all distances and average by the number of data points
    intraCluster = sum(distances, 'all') / size(dataPoints, 1);
end

function SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m)
    % U: N x c matrix of membership values for each data point in each cluster
    % dataPoints: N x D matrix of data points
    % clusterCenters: c x D matrix of cluster centers
    % m: Fuzziness exponent (usually > 1)


    % Intra-cluster distances (between data points and cluster centers)
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean'); % N x c matrix

    % Weight the distances using the membership values raised to the power m
    numerator = sum((U.^m) .* distances, 1); % 1 x c vector

    % Inter-cluster distances (between cluster centers)
    clusterDistances = pdist2(clusterCenters, clusterCenters, 'squaredeuclidean'); % c x c matrix

    % Add a small epsilon to avoid division by zero
    epsilon = 1e-10;

    % Sum of inter-cluster distances for each cluster center (sum of each row)
    % Number of data points
    N = size(dataPoints,1);
    denominator = N * sum(clusterDistances, 2)' + epsilon; % 1 x c vector

    % Compute the partition index (SC) for each cluster and sum them
    SC = sum(numerator ./ denominator);
end


function PC = calculatePartitionCoefficient(U)
    % U: N x c matrix of membership values for each data point in each cluster
    
    % Get the number of data points (N)
    N = size(U, 1);
    
    % Calculate the partition coefficient
    PC = sum(U.^2, 'all') / N;
end
function CE = calculateClassificationEntropy(U)
    % U: c x N matrix of membership values for each data point in each cluster
    
    % Get the number of data points (N)
    N = size(U, 1);
    
    % Avoid log(0) by adding a very small value (epsilon) to U
    epsilon = 1e-10;
    
    % Compute the classification entropy
    CE = -sum(U .* log(U + epsilon), 'all') / N;
end

function S = fuzzySeparationIndex(data, centroids, U, m)
    % data: matrix of size [num_samples, num_features]
    % centroids: matrix of size [num_clusters, num_features]
    % U: N x c matrix of membership values for each data point in each cluster
    % m: fuzziness exponent (typically m > 1)
    
    % Number of samples and clusters
    [num_samples, ~] = size(data);
    
    % Calculate the numerator
    % Compute the squared Euclidean distances in a vectorized way
    dist_matrix = pdist2(data, centroids, 'euclidean').^2; % N x c matrix of squared distances
    U_m = U'.^m; % Raise membership values to power m
    numerator = sum(sum(U_m .* dist_matrix)); % Compute the weighted sum of distances
    
    % Calculate the denominator
    % Compute pairwise squared distances between centroids
    centroid_distances = pdist(centroids, 'euclidean').^2; % Vector of pairwise squared distances
    min_inter_centroid_dist = min(centroid_distances); % Find the minimum non-zero distance
    
    % Calculate the separation index
    denominator = num_samples * min_inter_centroid_dist;
    S = numerator / denominator;
end

function varargout = customFCM(data, options)
%FCM Data set clustering using fuzzy c-means clustering.
%
%   [CENTER, FUZZYPARTMAT, OBJFCN, INFO] = FCM(DATA,OPTIONS) finds
%   clusters in the data set DATA using options set OPTIONS. DATA size is M-by-N, 
%   where M is the number of data points and N is the number of
%   coordinates/features for each data point. If the number
%   of clusters in OPTIONS is set to 'auto', or it is specified as a vector
%   of multiple cluster numbers, the outputs CENTER, FUZZYPARTMAT, and
%   OBJFCN represent the results for the optimal number of clusters.
%
%   CENTER is a K-by-N matrix where K is the optimal cluster number in the
%   auto-generated cluster numbers. The coordinates for each cluster center
%   are returned in the rows of the matrix CENTER.
% 
%   The membership function matrix FUZZYPARTMAT contains the grade of
%   membership of each DATA point in each cluster. The values 0 and 1
%   indicate no membership and full membership, respectively. Grades between
%   0 and 1 indicate that the data point has partial membership in a
%   cluster. 
% 
%   At each iteration, an objective function is minimized to find
%   the best location for the clusters and its values are returned in
%   OBJFCN.
%
%   Output INFO includes results for each cluster number. INFO is a
%   structure having the following fields:
%       NumClusters         : Auto-generated or user-specified cluster
%                             numbers
%       ClusterCenters      : Cluster centers generated in each FCM
%                             clustering with a different cluster number
%       FuzzyPartitionMatrix: Fuzzy partition matrix generated in each FCM
%                             clustering with a different cluster number
%       ObjectiveFcnValue   : Objective function values generated in each
%                             FCM clustering with a different cluster
%                             number
%       ValidityIndex       : Cluster validity measure/index values
%                             generated in each FCM clustering with a
%                             different cluster number
%       OptimalNumClusters  : Optimal number of clusters K corresponding to
%                             the minimum validity index value
%   For 'mahalanobis'and 'fmle' distance metric (specified in FCMOPTIONS),
%   INFO also includes CovarianceMatrix field that includes covariance
%   matrices of each cluster generated for a specific cluster number.
%
%   Example
%       data = rand(100,2);
%       options = fcmOptions();
%       [center,fuzzyPartMat,objFcn] = fcm(data,options);
%       plot(data(:,1), data(:,2),'o');
%       hold on;
%       maxU = max(U);
%       % Find the data points with highest grade of membership in cluster 1
%       index1 = find(U(1,:) == maxU);
%       % Find the data points with highest grade of membership in cluster 2
%       index2 = find(U(2,:) == maxU);
%       line(data(index1,1),data(index1,2),'marker','*','color','g');
%       line(data(index2,1),data(index2,2),'marker','*','color','r');
%       % Plot the cluster centers
%       plot([center([1 2],1)],[center([1 2],2)],'*','color','k')
%       hold off;
%
%   See also EUCLIDEANDIST, MAHALANOBISDIST, FMLE

%   Copyright 2022-2023 The MathWorks, Inc.

dataSize = size(data, 1);
objFcn = zeros(options.MaxNumIteration, 1);        % Array for objective function
iterationProgressFormat = getString(message('fuzzy:general:lblFcm_iterationProgressFormat'));

maxAutoK = 10;
if isequal(options.NumClusters,fuzzy.clustering.FCMOptions.DefaultNumClusters)
    if dataSize>2
        if isempty(options.ClusterCenters)
            kStart = 2;
        else
            kStart = size(options.ClusterCenters,1);
        end
        maxNumClusters = min(kStart+maxAutoK-1,size(data,1)-1);
        kValues = [kStart kStart+1:maxNumClusters];
    else
        if isempty(options.ClusterCenters)
            kValues = dataSize;
        else
            kValues = size(options.ClusterCenters,1);
        end
    end
else
    kValues = options.NumClusters;
end
lastResults = [];
centers = options.ClusterCenters;
validityIndexOutput = zeros(numel(kValues),2);
ct = 0;
numFolds = numel(kValues);
infoVal = {cell(1,numFolds)};
info = struct(...
    'NumClusters',zeros(1,numFolds), ...
    'ClusterCenters',infoVal, ...
    'FuzzyPartitionMatrix',infoVal, ...
    'ObjectiveFcnValue',infoVal, ...
    'CovarianceMatrix',infoVal, ...
    'ValidityIndex',zeros(1,numFolds), ...
    'OptimalNumClusters',0 ...
    );
minKIndex = Inf;
for k = kValues
    ct = ct + 1;
    
    options.ClusterCenters = [];
    options.NumClusters = k;
    if numel(options.ClusterVolume)~=k 
        if isscalar(options.ClusterVolume)
            options.ClusterVolume = options.ClusterVolume(1,ones(1,k));
        else
            options.ClusterVolume = ones(1,k);
        end
    end
    numCenters = size(centers,1);
    if numCenters>0
        if numCenters<k
            options.ClusterCenters = fuzzy.clustering.addKMPPCenters(data, ...
                centers,k);
        else
            options.ClusterCenters = centers;
        end
    end
    fuzzyPartMat = fuzzy.clustering.initfcm(options,dataSize);  % Initial fuzzy partition matrix
    for iterId = 1:options.MaxNumIteration
        [center, fuzzyPartMat, objFcn(iterId), covMat, stepBrkCond] = stepfcm(data,fuzzyPartMat,options);

        % Check termination condition
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
    validityIndex = fuzzy.clustering.clusterValidityIndex(data,center,fuzzyPartMat);

    if options.Verbose && ~isscalar(kValues)
        fprintf(...
            getString(message('fuzzy:general:msgFCM_currNumClusterAndValidity')),...
            k,validityIndex);
    end

    if isempty(lastResults) || validityIndex<lastResults.validityIndex
        lastResults.center = center;
        lastResults.fuzzyPartMat = fuzzyPartMat;
        lastResults.objFcn = objFcn;
        lastResults.covMat = covMat;
        lastResults.validityIndex = validityIndex;
        lastResults.k = k;
        minKIndex = ct;
    end
    if options.Verbose && ~isscalar(kValues)
        fprintf(...
            getString(message('fuzzy:general:msgFCM_optNumClusterAndValidity')), ...
            lastResults.k,lastResults.validityIndex);
        fprintf('\n');
    end
    if ~isempty(options.ClusterCenters)
        centers = lastResults.center;
    end
    validityIndexOutput(ct,:) = [k validityIndex];

    info.NumClusters(ct) = k;
    info.ClusterCenters{ct} = center;
    info.FuzzyPartitionMatrix{ct} = fuzzyPartMat;
    info.ObjectiveFcnValue{ct} = objFcn;
    info.CovarianceMatrix{ct} = covMat;
    info.ValidityIndex(ct) = validityIndex;
    
end
lastResults.objFcn = objFcn;
lastResults.validityIndexOutput = validityIndexOutput;
info.OptimalNumClusters = lastResults.k;
if strcmp(options.DistanceMetric,getString(message('fuzzy:general:lblFcm_euclidean')))
    info = rmfield(info,"CovarianceMatrix");
end

[varargout{1:nargout}] = assignOutputs(info,minKIndex);
end

%% Local functions
function brkCond = checkBreakCondition(options,objFcn,iterId,stepBrkCond)
%%

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
if iterId==options.MinNumIteration
    brkCond.isTrue = false;
end

end

function varargout = assignOutputs(info,minIndex)
%%

if nargout>3
    varargout{4} = info;
end
if nargout>2
    varargout{3} = info.ObjectiveFcnValue{minIndex};
end
if nargout>1
    varargout{2} = info.FuzzyPartitionMatrix{minIndex};
end
if nargout>0
    varargout{1} = info.ClusterCenters{minIndex};
end
end

function [center, newFuzzyPartMat, objFcn, covMat, brkCond] = stepfcm(data, fuzzyPartMat, options)
%STEPFCM One step in fuzzy c-mean clustering with combined objective function.
%
%   [CENTER, NEWFUZZYPARTMAT, OBJFCN, COVMAT, BRKCOND] = STEPFCM(DATA, FUZZYPARTMAT, OPTIONS)
%   performs one iteration of fuzzy c-mean clustering with an additional
%   custom fitness function incorporated into the objective.

numCluster = options.NumClusters;
expo = options.Exponent;
%lambda = options.lambda; % Weight for the custom fitness function
lambda = options.lambda;
clusterVolume = options.ClusterVolume;
brkCond = struct('isTrue', false, 'description', '');

memFcnMat = fuzzyPartMat.^expo; % Fuzzy Partition Matrix after exponential modification
if isempty(options.ClusterCenters)
    center = memFcnMat * data ./ (sum(memFcnMat, 2) * ones(1, size(data, 2))); % New cluster centers
else
    center = options.ClusterCenters;
    if options.UsePerturbation
        center = center + options.PerturbationFactor * randn(size(center));
    end
    options.ClusterCenters = [];
end

% Calculate distances based on the chosen distance metric
if strcmp(options.DistanceMetric, getString(message('fuzzy:general:lblFcm_mahalanobis')))
    [dist, covMat, brkCond] = fuzzy.clustering.mahalanobisdist(center, data, memFcnMat, clusterVolume);
elseif strcmp(options.DistanceMetric, getString(message('fuzzy:general:lblFcm_fmle')))
    [dist, covMat, brkCond] = fuzzy.clustering.fmle(center, data, memFcnMat);
else
    dist = fuzzy.clustering.euclideandist(center, data);
    covMat = [];
end

% Calculate the traditional FCM objective function
fcmObjective = sum(sum((dist.^2) .* max(memFcnMat, eps)));

% Calculate the custom fitness value
fitnessValue = calculateFitness(center, options);
objFcn = calculateFitness(center, options);

% Combine the traditional FCM objective and fitness function
%vobjFcn = fcmObjective + lambda * fitnessValue;

% Update the fuzzy partition matrix
tmp = (max(dist, eps)).^(-2/(expo-1)); % Calculate new Fuzzy Partition Matrix, suppose expo != 1
newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));

end
