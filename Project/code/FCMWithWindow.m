clearvars;

load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions


[r,c,n,~] = size(D);
DTrain = reshape(D(:,:,:,1),[r*c n]);
kDim = [3 3];
DTrainFeatures = createMovingWindowFeatures(DTrain,kDim);
%%
nClusters = 15;

options = struct();
options.NumClusters = nClusters;
options.ClusterCenters = [];
options.Exponent = 2;
options.MaxNumIteration = 10;
options.DistanceMetric = 'euclidean';
options.MinImprovement = 1e-5;
options.Verbose = 1;
options.ClusterVolume = 1;
options.lambda = 0;

[fcmCenters, fcmU] = customFCM(DTrainFeatures, options);
%%
ecDist = findDistance(fcmCenters,DTrainFeatures);
[~,ecLabel] = min(ecDist',[],2); %#ok<UDIM>
ecLabel = reshape(ecLabel,n,r*c)';
ecLabel = reshape(ecLabel,[r c n]);
figure
subplot(1,2,1)
slice_number = 15;
sliceData = D(:,:,slice_number);
imshow(sliceData)
xlabel("Orig Image")
subplot(1,2,2)
imshow(ecLabel(:,:,slice_number)/nClusters)
xlabel("Segmented Image")
%%
options = struct();
options.NumClusters = nClusters;
options.ClusterCenters = [];
options.Exponent = 2;
options.MaxNumIteration = 10;
options.DistanceMetric = 'euclidean';
options.MinImprovement = 1e-5;
options.Verbose = 1;
options.ClusterVolume = 1;
options.lambda = 0;

% BAT Options.
options.nBats = 50;
options.BATIterMax = 10;
options.lowerBound = min(DTrainFeatures);
options.upperBound = max(DTrainFeatures);
options.Qmin = 0;
options.Qmax = 2;
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.gamma = 0.95; % Decay rate for pulse rate
options.chaotic = false;
options.MinNumIteration = 50;
options.UsePerturbation = true;
options.PerturbationFactor = 0.01;

% Apply BAT + Fuzzy C-Means (FCM) clustering
segImgInresults = MFBAFCM(DTrainFeatures, options);
%%
ecDist = findDistance(segImgInresults.centers,DTrainFeatures);
[~,ecLabel] = min(ecDist',[],2); %#ok<UDIM>
ecLabel = reshape(ecLabel,n,r*c)';
ecLabel = reshape(ecLabel,[r c n]);
figure
subplot(1,2,1)
slice_number = 15;
sliceData = D(:,:,slice_number);
imshow(sliceData)
xlabel("Orig Image")
subplot(1,2,2)
imshow(ecLabel(:,:,slice_number)/nClusters)
xlabel("Segmented Image")

%%
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
%fcmObjective = sum(sum((dist.^2) .* max(memFcnMat, eps)));

% Calculate the custom fitness value
objFcn = calculateFitness(center, data, options);

% Combine the traditional FCM objective and fitness function
%vobjFcn = fcmObjective + lambda * fitnessValue;

% Update the fuzzy partition matrix
tmp = (max(dist, eps)).^(-2/(expo-1)); % Calculate new Fuzzy Partition Matrix, suppose expo != 1
newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));

end

%%
function fitness = calculateFitness(clusterCenters, data, options)
    % dataPoints: N×D (number of data points × number of features).
    % clusterCenters: C×D (number of clusters × number of features).

    % Extract the data points and number of clusters from options
    m = options.Exponent; % Fuzziness exponent

    % Calculate intra-cluster distance
    intra_cluster = calculateIntraCluster(data, clusterCenters); % No transpose needed

    % Compute distances between data points and cluster centers (N x C)
    distances = pdist2(data, clusterCenters).^2; % Squared distances

    % Avoid division by zero by setting very small values to a small epsilon
    epsilon = 1e-10;
    distances(distances < epsilon) = epsilon;

    % Update membership matrix U using vectorized operations
    exponent = 2 / (m - 1);
    invDistances = 1 ./ distances; % Element-wise inversion of distances
    sumInvDistances = sum(invDistances .^ exponent, 2); % Sum across clusters for each data point

    U = (invDistances .^ exponent) ./ sumInvDistances; % Calculate membership values

    % Calculate partition index (SC)
    SC = calculatePartitionIndex(U, data, clusterCenters, m);

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
%
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
    for i = 1:options.nBats
        % Reshape each bat into a (numClusters x calculateFitness) format for calculateFitness
        reshapedBat = reshape(bats(i, :), [options.NumClusters, numFeatures]);
        fitness(i) = calculateFitness(reshapedBat, data, options);  % Calculate fitness based on clustering
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestClusterCenters = bats(idx, :);  % Store best solution as flattened vector

    % Main loop for the BAT algorithm
    for t = 1:options.BATIterMax
        fprintf("Iter:%d\n", t);
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
            reshapedNewSolution = reshape(newClusterCenters, [options.NumClusters, numFeatures]);
            newFitness = calculateFitness(reshapedNewSolution, data, options);

            % Acceptance criteria based on fitness, loudness, and pulse rate
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                bats(i, :) = newClusterCenters;
                fitness(i) = newFitness;
                loudnesses(i) = options.loudnessCoefficient * loudnesses(i);  % Decrease loudness
                pulseRates(i) = pulseRates(i) * (1 - exp(-options.gamma * t));  % Increase pulse rate
            end

            % Update global best if a better solution is found
            if newFitness < bestFitness
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
    % Reshape to (numClusters x 9) for easy boundary enforcement
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