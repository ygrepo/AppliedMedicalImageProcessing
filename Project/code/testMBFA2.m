% MFBA Algorithm with Segmentation for Brain Tumor Detection (Using MATLAB's Built-In MRI Data)

% Parameters initialization
num_bats = 20; % Number of bats for MFBA
max_iterations = 100; % Maximum number of MFBA iterations
num_clusters = 3; % Number of clusters for segmentation
alpha = 0.9; 
gamma = 0.9;
fmin = 0; 
fmax = 2;
m = 2; % Fuzziness parameter for FCM

% Chaotic map (sinusoidal as an example)
chaotic_map = @(x) sin(2 * pi * x);

% Load the built-in MRI dataset
load('mri.mat');
mriData = squeeze(D); % Extract the MRI data and remove singleton dimensions
sliceNumber = 13; % Select a slice to use for segmentation
grayImage = double(mriData(:,:,sliceNumber)); % Select the slice and convert to double
dataPoints = grayImage(:); % Reshape image into a vector

% Initialize MFBA positions (cluster centers)
positions = rand(num_bats, num_clusters, 1); % Random initial cluster centers
velocities = zeros(num_bats, num_clusters, 1); % Initialize velocities for MFBA

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

        % Perform Fuzzy C-Means Clustering to get the cluster centers and membership matrix
        [clusterCenters, U] = fcm(dataPoints, num_clusters);
        
        % Calculate intra_cluster, SC, and PC using the cluster centers from FCM
        intra_cluster = calculateIntraCluster(dataPoints, clusterCenters, U);
        SC = calculateSC(dataPoints, clusterCenters, U, m);
        PC = calculatePC(U);
        
        % Fitness function as per Equation (21)
        fitness = (intra_cluster + SC) / PC;
        
        % Update global best
        if fitness < global_best_fitness && chaotic_value < loudness(i)
            positions(i, :, :) = new_position;
            global_best_fitness = fitness;
            global_best_centers = clusterCenters; % Use FCM's cluster centers
            loudness(i) = alpha * loudness(i); % Reduce loudness
            pulse_rate(i) = gamma * (1 - exp(-iter)); % Increase pulse rate
        end
    end
end


% Use global_best_centers for segmentation
best_centers = squeeze(global_best_centers);

% Compute the distances of each pixel to the cluster centers
distances = zeros(numel(dataPoints), num_clusters);
for j = 1:num_clusters
    distances(:, j) = abs(dataPoints - best_centers(j));
end

% Assign each pixel to the nearest cluster
[~, labels] = min(distances, [], 2);
segmented_image = reshape(labels, size(grayImage));

% Identify the tumor cluster (e.g., the cluster with the highest intensity)
tumor_cluster = find(best_centers == max(best_centers));
tumor_segment = segmented_image == tumor_cluster;

% Display results
figure;
subplot(1, 2, 1);
imshow(mriData(:,:,sliceNumber), []);
title('Original MRI Image Slice');

subplot(1, 2, 2);
imshow(tumor_segment, []);
title('Segmented Tumor');

function intra_cluster = calculateIntraCluster(dataPoints, clusterCenters, U)
    % dataPoints: N x 1 matrix (N data points, 1 feature for grayscale intensities)
    % clusterCenters: c x 1 matrix (c cluster centers, 1 feature)
    % U: N x c matrix of membership values for each data point in each cluster

    [N, ~] = size(dataPoints);
    [c, ~] = size(clusterCenters);
    
    % Ensure U and clusterCenters have the correct size
    assert(c == size(U, 1), 'The number of clusters in U and clusterCenters must match.');
    
    intra_cluster = 0;
    
    for i = 1:N
        for j = 1:c
            % Adjusted for 1D data points and cluster centers
            intra_cluster = intra_cluster + (U(i, j) ^ 2) * (dataPoints(i) - clusterCenters(j))^2;
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