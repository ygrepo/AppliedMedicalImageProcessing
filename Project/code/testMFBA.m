% Initialization
clearvars
num_bats = 20; % Number of bats for MFBA
max_iterations = 100; % Maximum number of MFBA iterations
num_clusters = 3; % Number of clusters for segmentation

% Define MFBA parameters, chaotic maps, etc. (from previous MFBA code)
alpha = 0.9; 
gamma = 0.9;
fmin = 0; 
fmax = 2;
accuracy_weight = 0.8;
chaotic_map = @(x) sin(2 * pi * x);

% Load and preprocess the brain image data (as in the original example)
dataDir = "/Users/yvesgreatti/github/AppliedMedicalImageProcessing/Project/data/";
imageDir = fullfile(dataDir,"BraTS");
% filename = matlab.internal.examples.downloadSupportFile(...
%     "vision","data/sampleBraTSTestSetValid.tar.gz");
% untar(filename,imageDir);

trainDataFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","imagesTest","BraTS447.mat");
testDataFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","imagesTest","BraTS463.mat");
testLabelFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","labelsTest","BraTS463.mat");

orgTrainData = load(trainDataFileName);
[r, c, n, ~] = size(orgTrainData.cropVol);
trainingData = reshape(orgTrainData.cropVol(:,:,:,1), [r*c n]);
%%
% Feature extraction (similar to original example)
kDim = [3 3];
trainFeatures = createMovingWindowFeatures(trainingData, kDim);
%%
% Initialize MFBA positions (cluster centers)
positions = rand(num_bats, num_clusters, size(trainFeatures, 2)); % Random initial cluster centers
global_best_fitness = inf;
global_best_centers = positions(1, :, :);

% MFBA Main Loop
for iter = 1:max_iterations
    chaotic_value = chaotic_map(mod(iter, max_iterations) / max_iterations);
    
    for i = 1:num_bats
        % Frequency adjustment
        frequencies(i) = fmin + (fmax - fmin) * chaotic_value;
        
        % Update velocities and positions (cluster centers)
        velocities(i, :, :) = velocities(i, :, :) + (positions(i, :, :) - global_best_centers) * frequencies(i);
        new_position = positions(i, :, :) + velocities(i, :, :);
        
        % Apply chaotic perturbation to cluster centers
        new_position = new_position + chaotic_value * (chaotic_map(chaotic_value) - 0.5);
        
        % Evaluate fitness using clustering (e.g., using intra-cluster distance)
        intra_cluster = calculateIntraCluster(trainFeatures, new_position); % Implement this function as per Equation 22
        PC = sum(sum(U.^2)) / size(trainFeatures, 1); % Partition coefficient (needs implementation)
        SC = sum(sum(U.^2 .* intra_cluster)); % Partition index (simplified)

        fitness = (intra_cluster + SC) / PC; % Fitness function as per Equation 21

        % Update global best
        if fitness < global_best_fitness
            global_best_fitness = fitness;
            global_best_centers = new_position;
        end
    end
end

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

function dist = findDistance(centers,data)
% Calculate feature distance from cluster center.

dist = zeros(size(centers, 1), size(data, 1));
for k = 1:size(centers, 1)
    dist(k, :) = sqrt(sum(((data-ones(size(data, 1), 1)*centers(k, :)).^2), 2));
end
end

function intra_cluster = calculateIntraCluster(dataPoints, clusterCenters, U)
    % dataPoints: N x D matrix (N data points, D features)
    % clusterCenters: c x D matrix (c cluster centers, D features)
    % U: N x c matrix of membership values for each data point in each cluster

    [N, ~] = size(dataPoints);
    [c, ~] = size(clusterCenters);
    
    intra_cluster = 0;
    
    for i = 1:N
        for j = 1:c
            intra_cluster = intra_cluster + (U(i, j) ^ 2) * norm(dataPoints(i, :) - clusterCenters(j, :))^2;
        end
    end
    
    intra_cluster = intra_cluster / N;
end

function SC = calculateSC(dataPoints, clusterCenters, U, m)
    % dataPoints: N x D matrix (N data points, D features)
    % clusterCenters: c x D matrix (c cluster centers, D features)
    % U: N x c matrix of membership values for each data point in each cluster
    % m: Fuzziness parameter (typically m > 1)

    [N, ~] = size(dataPoints);
    [c, ~] = size(clusterCenters);
    
    % Calculate the numerator (intra-cluster distances)
    numerator = 0;
    for j = 1:c
        for i = 1:N
            numerator = numerator + (U(i, j) ^ m) * norm(dataPoints(i, :) - clusterCenters(j, :))^2;
        end
    end
    
    % Calculate the denominator (distance between cluster centers)
    denominator = 0;
    for j = 1:c
        for k = 1:c
            denominator = denominator + norm(clusterCenters(k, :) - clusterCenters(j, :))^2;
        end
    end
    
    SC = numerator / (N * denominator);
end
function PC = calculatePC(U)
    % U: N x c matrix of membership values for each data point in each cluster

    [N, c] = size(U);
    
    % Calculate the partition coefficient
    PC = sum(sum(U.^2)) / N;
end
