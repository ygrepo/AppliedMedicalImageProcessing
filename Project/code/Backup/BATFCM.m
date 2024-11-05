% Load an example MRI image
clearvars 
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Select the 15th slice of the MRI
slice_number = 15;
sliceData = D(:,:,slice_number);  % uint8 data
sliceData = histeq(sliceData);

% Convert uint8 data to double and normalize to [0, 1]
data = double(sliceData(:)) / 255;  % Normalizing the pixel values

nClusters = 4;

fcmOpt = fcmOptions(NumClusters = nClusters);
disp('FCM Options')
disp(fcmOpt)

% Apply Fuzzy C-Means (FCM) clustering
[fcmCenters, fcmU] = fcm(data, fcmOpt);

% Get the cluster membership for each pixel
[~, maxU] = max(fcmU);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segFCMImg = reshape(maxU, size(sliceData));



% Example parameters
options = struct();
options.nBats = 50;
options.itermax = 100;
options.lowerBound = min(data);
options.upperBound = max(data);
options.nClusters = nClusters;
options.m = 2; % Fuzziness exponent
options.Qmin = 0;
options.Qmax = 2;
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.gamma = 0.95; % Decay rate for pulse rate
options.chaotic = false;

options.epsilon = 1e-6; % Convergence criterion for FCM
options.fcmIterMax = 300;
options.DistanceMetric = 'euclidean';
options.dataPoints = data;


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
    fcmCenters, fcmOpt.Exponent);
S = fuzzySeparationIndex(data, fcmCenters,...
    fcmU, fcmOpt.Exponent);
fprintf("FCM: PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f\n", PC,CE,SC, S);
batFCMUT = segImgInresults.U';
PC = calculatePartitionCoefficient(batFCMUT);
CE = calculateClassificationEntropy(batFCMUT);
SC = calculatePartitionIndex(batFCMUT, data, ...
    segImgInresults.centers, options.m);
S = fuzzySeparationIndex(data, segImgInresults.centers,...
    segImgInresults.U, options.m);
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
%%
function [bestSol, bestFitness] = batAlgorithm(options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rates
    loudnesses = options.loudness * ones(options.nBats, 1);  % Per-bat loudness

    % Initialize bat positions for cluster centers and velocities
    bats = options.lowerBound + (options.upperBound - options.lowerBound) ...
        .* rand(options.nBats, options.nClusters);
    velocities = zeros(options.nBats, options.nClusters);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats
    for i = 1:options.nBats
        fitness(i) = calculateFitness(bats(i, :)', options);  % Call to calculateFitness
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestSol = bats(idx, :);

    % Main loop for the BAT algorithm
    for t = 1:options.itermax
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            Q = options.Qmin + (options.Qmax - options.Qmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestSol) * Q;
            newSolution = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints
            newSolution = enforceBoundaries(newSolution, options.lowerBound, options.upperBound);

            % Local search
            if rand > pulseRates(i)
                newSolution = bestSol + 0.01 * randn(1, options.nClusters);
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

    if isfield(options, 'dataPoints')

        % Extract the data points and number of clusters from options
        dataPoints = options.dataPoints;
        m = options.m; % Fuzziness exponent
    
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
%%
function [clusterCenters, U] = customFCM(dataPoints, nClusters, options, initialCenters)
    % Get the number of data points
    N = length(dataPoints);
    m = options.m; % Fuzziness exponent
    itermax = options.fcmIterMax; % Maximum number of iterations
    epsilon = options.epsilon; % Convergence criterion

    % Initialize cluster centers
    clusterCenters = initialCenters;

    % Initialize the membership matrix (U)
    U = rand(nClusters, N);
    U = U ./ sum(U, 1); % Normalize to ensure the sum of memberships for each data point is 1

    % Iterate until convergence or max iterations
    for iter = 1:itermax
        U_prev = U; % Store the previous membership matrix

        % Update cluster centers
        for j = 1:nClusters
            numerator = sum((U(j, :) .^ m) .* dataPoints');
            denominator = sum(U(j, :) .^ m);
            clusterCenters(j) = numerator / denominator;
        end

        % Update membership matrix U
        for i = 1:N
            for j = 1:nClusters
                dist = abs(dataPoints(i) - clusterCenters(j));
                if dist == 0
                    U(:, i) = 0;
                    U(j, i) = 1;
                else
                    U(j, i) = 1 / sum((dist / abs(dataPoints(i) - clusterCenters(:))).^(2 / (m - 1)));
                end
            end
        end

        % Check for convergence
        if max(max(abs(U - U_prev))) < epsilon
            break;
        end
    end
end
