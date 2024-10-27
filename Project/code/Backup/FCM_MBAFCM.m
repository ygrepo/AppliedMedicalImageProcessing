clear all; clc;

% Load the MRI dataset
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Display the 15th slice of the MRI
slice_number = 15;
figure;
imshow(D(:,:,slice_number), []);
title(['Slice ', num2str(slice_number)]);

%% 

%%% FCM 
% Reshape the 15th slice to be a 2D array (flatten the image)
slice_data = D(:,:,slice_number);
pixel_data = double(slice_data(:));

% Number of clusters for segmentation (e.g., 3 clusters for tissue types)
n_clusters = 3;

% Initialize Fuzzy C-Means clustering
[center, U] = fcm(pixel_data, n_clusters);

% Find the highest membership cluster for each pixel
[~, cluster_index] = max(U);

% Reshape the clustered data back to the image size
segmented_image = reshape(cluster_index, size(slice_data));

% Display the segmented image
figure;
imshow(segmented_image, []);
title('Segmented MRI Slice Using Fuzzy C-Means');

%% 

%%% MBA and FCM

% Step 1: Load MRI Data and Initialize Parameters
load mri;  % Load preloaded MRI dataset
D = squeeze(D);  % Removes singleton dimensions

% Parameters for MFBA and FCM
n_bats = 20;            % Number of bats in the population
n_clusters = 3;         % Number of clusters for FCM
n_iterations = 100;     % Maximum iterations
fmin = 1;               % Minimum frequency
fmax = 2;               % Maximum frequency
A = 0.9;                % Loudness
r0 = 0.1;               % Initial pulse emission rate
alpha = 0.9;            % Loudness damping factor
gamma = 0.9;            % Pulse rate increase factor

% Load the MRI slice for segmentation
slice_number = 15;
slice_data = double(D(:,:,slice_number));  % Convert to double for precision
pixel_data = slice_data(:);  % Flatten the image to a 1D array
n_pixels = length(pixel_data);

% Initialize the bat population (each bat holds cluster centers)
bats_position = rand(n_bats, n_clusters) * (max(pixel_data) - min(pixel_data)) + min(pixel_data);
bats_velocity = zeros(n_bats, n_clusters);

% Initialize loudness and pulse rate for each bat
loudness = A * ones(n_bats, 1);
pulse_rate = r0 * ones(n_bats, 1);

% Best solution initialization
best_solution = bats_position(1, :);  % Initialize best solution with the first bat
best_fitness = Inf;  % Start with an infinitely bad fitness value

% Step 2: Fitness Function
function fitness = calculate_fitness(cluster_centers, pixel_data, n_clusters)
    % Perform FCM with the given cluster centers
    [~, U] = fcm_custom(pixel_data, n_clusters, cluster_centers);

    % Calculate intra-cluster distance as part of fitness
    intra_cluster_dist = 0;
    for i = 1:n_clusters
        cluster_points = pixel_data(U(i,:) == max(U));
        intra_cluster_dist = intra_cluster_dist + sum((cluster_points - cluster_centers(i)).^2);
    end

    % The fitness is based on the intra-cluster distance
    fitness = intra_cluster_dist;  % A lower value indicates better clustering
end


% Step 3: MFBA Iterations to Optimize FCM Centers
for iter = 1:n_iterations
    for i = 1:n_bats
        % Update frequency for each bat
        freq = fmin + (fmax - fmin) * rand;
        
        % Update velocity and position of each bat
        bats_velocity(i, :) = bats_velocity(i, :) + (bats_position(i, :) - best_solution) * freq;
        bats_position(i, :) = bats_position(i, :) + bats_velocity(i, :);
        
        % Generate a new local solution if the random number is greater than the pulse rate
        if rand > pulse_rate(i)
            bats_position(i, :) = best_solution + 0.1 * randn(1, n_clusters);  % Random walk around the best solution
        end
        
        % Apply FCM with the new bat position (cluster centers)
        fitness = calculate_fitness(bats_position(i, :), pixel_data, n_clusters);
        
        % Accept the new solution if it improves the fitness
        if fitness < best_fitness && rand < loudness(i)
            best_solution = bats_position(i, :);
            best_fitness = fitness;
            
            % Update loudness and pulse rate
            loudness(i) = alpha * loudness(i);
            pulse_rate(i) = r0 * (1 - exp(-gamma * iter));
        end
    end
end


% Step 4: Final FCM Segmentation Using Optimized Cluster Centers
[~, U] = fcm_custom(pixel_data, n_clusters, best_solution);

% Convert the cluster memberships into segmented image
[~, cluster_index] = max(U);
segmented_image = reshape(cluster_index, size(slice_data));

% Display the segmented MRI slice
figure;
imshow(segmented_image, []);
title('Segmented MRI Slice Using MFBAFCM');


% Step 5: Custom FCM Function with Initialized Centers
function [center, U] = fcm_custom(pixel_data, n_clusters, init_centers)
    % Initialize cluster centers
    center = init_centers;
    
    % Set fuzziness parameter and stopping criteria
    max_iter = 100;
    epsilon = 1e-5;
    m = 2;  % Fuzziness parameter

    % Initialize the membership matrix randomly
    U = rand(n_clusters, length(pixel_data));
    U = U ./ sum(U, 1);  % Normalize memberships
    
    for iter = 1:max_iter
        % Update cluster centers
        for j = 1:n_clusters
            numerator = sum((U(j,:).^m) .* pixel_data');
            denominator = sum(U(j,:).^m);
            center(j) = numerator / denominator;
        end
        
        % Update membership matrix U
        for i = 1:length(pixel_data)
            for j = 1:n_clusters
                sum_term = sum((pixel_data(i) - center(j)).^2 ./ ((pixel_data(i) - center).^2));
                U(j,i) = 1 / (sum_term^(1/(m-1)));
            end
        end
        
        % Check for convergence
        if max(abs(center(:) - init_centers(:))) < epsilon
            break;
        end
    end
end