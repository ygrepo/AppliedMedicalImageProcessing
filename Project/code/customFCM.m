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

% function [center, newFuzzyPartMat, objFcn, covMat, brkCond] = stepfcm(data, fuzzyPartMat, options)
% %STEPFCM One step in fuzzy c-mean clustering.
% %
% %   [CENTER, NEWFUZZYPARTMAT, OBJFCN, COVMAT, BRKCOND] = STEPFCM(DATA, FUZZYPARTMAT, OPTIONS)
% %   performs one iteration of fuzzy c-mean clustering, where
% %
% %   DATA: matrix of data to be clustered. (Each row is a data point.)
% %   FUZZYPARTMAT: partition matrix. (FUZZYPARTMAT(i,j) is the MF value of
% %   data j in cluster i.)
% %   OPTIONS: includes properties NUMCLUSTERS and EXPONENT
% %   NEWFUZZYPARTMAT: new partition matrix.
% %   CENTER: center of clusters. (Each row is a center.)
% %   OBJFCN: objective function for partition FUZZYPARTMAT.
% %   COVMAT: covariance matrix for Mahalanobis distance metric.
% %   BRKCOND: Flag for breaking iterations according to conditions.
% %
% %   Note that the situation of "singularity" (one of the data points is
% %   exactly the same as one of the cluster centers) is not checked.
% %   However, it hardly occurs in practice.
% %
% %     See also EUCLIDEANDIST, MAHALANOBISDIST, FMLE, STEPFCM, FCM
% 
% %  Copyright 2022-2023 The MathWorks, Inc.
% 
% numCluster = options.NumClusters;
% expo = options.Exponent;
% clusterVolume = options.ClusterVolume;
% brkCond = struct('isTrue',false,'description','');
% 
% memFcnMat = fuzzyPartMat.^expo;       % Fuzzy Partition Matrix after exponential modification
% if isempty(options.ClusterCenters)
%     center = memFcnMat*data./(sum(memFcnMat,2)*ones(1,size(data,2)));     % New cluster centers
% else
%     center = options.ClusterCenters;
%     options.ClusterCenters = [];
% end
% 
% if strcmp(options.DistanceMetric, getString(message('fuzzy:general:lblFcm_mahalanobis')))
%     [dist, covMat, brkCond] = fuzzy.clustering.mahalanobisdist(center, data, memFcnMat, clusterVolume);
% elseif strcmp(options.DistanceMetric, getString(message('fuzzy:general:lblFcm_fmle')))
%     [dist, covMat, brkCond] = fuzzy.clustering.fmle(center, data, memFcnMat);
% else
%     dist = fuzzy.clustering.euclideandist(center, data);
%     covMat = [];
% end
% 
% tmp = (max(dist,eps)).^(-2/(expo-1));     % Calculate new Fuzzy Partition Matrix, suppose expo != 1
% newFuzzyPartMat = tmp./(ones(numCluster, 1)*sum(tmp));
% objFcn = sum(sum((dist.^2).*max(memFcnMat,eps)));
% 
% end

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

% Combine the traditional FCM objective and fitness function
objFcn = fcmObjective + lambda * fitnessValue;

% Update the fuzzy partition matrix
tmp = (max(dist, eps)).^(-2/(expo-1)); % Calculate new Fuzzy Partition Matrix, suppose expo != 1
newFuzzyPartMat = tmp ./ (ones(numCluster, 1) * sum(tmp));
end

%%
function fitness = calculateFitness(clusterCenters, options)
    % dataPoints: N×D (number of data points × number of features).
    % clusterCenters: C×D (number of clusters × number of features).

    if isfield(options, 'dataPoints')

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
    else
        fitness = options.fitness(clusterCenters); % Call the provided fitness function
    end
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
