
% Objective function for non-clustering problems (adjust as needed)
function f = fitness(x)
    % If x is a vector, the sum of squares computes the generalized Sphere function
    f = sum((x - 2).^2); % Sphere function centered at 2
end

% Example usage for non-clustering (1D problem)
options = struct();
options.nBats = 20; % Number of bats in the population
options.itermax = 100; % Number of iterations
options.lowerBound = -10; % Lower bound of the search space
options.upperBound = 10; % Upper bound of the search space
options.nClusters = 1; % Dimensionality of the problem (1D)
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.Qmin = 0; % Minimum frequency
options.Qmax = 2; % Maximum frequency
options.gamma = 0.1;
options.fitness = @fitness; % Use the fitness function for Sphere

% Call the algorithm for 1D
[bestSol, bestFitness] = batAlgorithm(options);
fprintf('Best solution (1D) found: %s\n', mat2str(bestSol));
fprintf('Best fitness (1D) found: %.4f\n', bestFitness);

%% Iris Data Testing ----
% Read the Iris data (assumes the file contains 5 columns: 4 features + 1 species label)
iris = readtable('fisheriris.csv');

% Separate data by species
setosa = iris(1:50,:);
versicolor = iris(51:100,:);
virginica = iris(101:150,:);

% Define feature names
Characteristics = {'sepal length', 'sepal width', 'petal length', 'petal width'};

% Convert the species data to arrays for plotting (features only, no species label)
setosa = table2array(setosa(:, 1:end-1));  % Features only
versicolor = table2array(versicolor(:, 1:end-1));
virginica = table2array(virginica(:, 1:end-1));

% Define pairs of features to be plotted (for scatter plots)
pairs = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4];

% Create a figure for plotting feature pairs
h = figure;
for j = 1:6
    x = pairs(j, 1);  % Feature index for x-axis
    y = pairs(j, 2);  % Feature index for y-axis
    subplot(2, 3, j);  % Create subplots
    % Plot each species separately using 'hold on'
    hold on;
    plot(setosa(:, x), setosa(:, y), 'r.');  % Setosa in red
    plot(versicolor(:, x), versicolor(:, y), 'g.');  % Versicolor in green
    plot(virginica(:, x), virginica(:, y), 'b.');  % Virginica in blue
    xlabel(Characteristics{x});  % Label x-axis with feature name
    ylabel(Characteristics{y});  % Label y-axis with feature name
    hold off;
end

% Fuzzy C-Means Clustering ----
% Define FCM options
options = [2.0, 100, 1e-6, 0];  % [Exponent, MaxNumIteration, MinImprovement, DisplayInfo]

% Perform Fuzzy C-Means clustering on the 4 features of the Iris dataset
[centers, U] = fcm(table2array(iris(:, 1:4)), 3, options);

% Plot the cluster centers on top of the existing subplots
for i = 1:6
    subplot(2, 3, i);  % Select the corresponding subplot
    hold on;
    for j = 1:3  % There are 3 clusters
        x = pairs(i, 1);  % Feature index for x-axis
        y = pairs(i, 2);  % Feature index for y-axis
        % Plot the cluster centers
        plot(centers(j, x), centers(j, y), 'kx', 'MarkerSize', 12, 'LineWidth', 2);  % Cluster center as a black 'x'
        text(centers(j, x) + .1, centers(j, y) + .1, int2str(j), 'FontWeight', 'bold', 'FontSize', 12);  % Label the center
    end
    hold off;
end

%%
% Define Bat Algorithm options
options = struct();
options.nBats = 20;  % Number of bats
options.itermax = 100;  % Maximum iterations
options.lowerBound = lowerBound;  % Lower bound of search space
options.upperBound = upperBound;  % Upper bound of search space
options.nClusters = 3;  % Number of clusters
options.pulseRate = 0.5;  % Initial pulse rate
options.loudness = 0.5;  % Initial loudness
options.Qmin = 0;  % Minimum frequency
options.Qmax = 2;  % Maximum frequency
options.gamma = 0.1;  % Loudness coefficient
options.dataPoints = table2array(iris(:, 1:4));  % Use the Iris data (features only)

% Run the Bat Algorithm to find the cluster centers
[bestSol, bestFitness] = batAlgorithm(options);
centers = reshape(bestSol, [], options.nClusters);  % Reshape the solution to [4 features x 3 clusters]
%%
% Set the bounds based on the Iris dataset (4 features)
nDimensions = size(iris, 2) - 1;  % Number of features (4)
nClusters = 3;  % Number of clusters

% Define the lower and upper bounds for the 4D feature space
lowerBound = repmat(min(table2array(iris(:, 1:nDimensions))), nClusters);
upperBound = repmat(max(table2array(iris(:, 1:nDimensions))), nClusters);


%% Plot the cluster centers on top of the existing subplots
for i = 1:6
    subplot(2, 3, i);  % Select the corresponding subplot
    hold on;
    for j = 1:options.nClusters  % There are 3 clusters
        x = pairs(i, 1);  % Feature index for x-axis
        y = pairs(i, 2);  % Feature index for y-axis
        % Plot the cluster centers
        plot(centers(x, j), centers(y, j), 'kx', 'MarkerSize', 12, 'LineWidth', 2);  % Cluster center as a black 'x'
        text(centers(x, j), centers(y, j), int2str(j), 'FontWeight', 'bold', 'FontSize', 12);  % Label the center
    end
    hold off;
end

%%
% Define 1D data points
dataPoints = [1.2, 1.5, 2.0, 8.5, 9.0, 9.2, 15.0, 16.5, 17.2];

% Number of clusters
nClusters = 3;

% Define the bounds for the 1D space (min and max of data points)
lowerBound = min(dataPoints);
upperBound = max(dataPoints);

% Define Bat Algorithm options
options = struct();
options.nBats = 20;  % Number of bats
options.itermax = 100;  % Maximum iterations
options.lowerBound = repmat(lowerBound, 1, nClusters);  % Lower bound for each cluster center
options.upperBound = repmat(upperBound, 1, nClusters);  % Upper bound for each cluster center
options.nClusters = nClusters;  % Number of clusters
options.pulseRate = 0.5;  % Initial pulse rate
options.loudness = 0.5;  % Initial loudness
options.Qmin = 0;  % Minimum frequency
options.Qmax = 2;  % Maximum frequency
options.gamma = 0.1;  % Loudness coefficient
options.dataPoints = dataPoints';  % Use the 1D data points
options.loudnessCoefficient = .9;
options.m = 2;

% Run the Bat Algorithm to find the cluster centers
[bestSol, bestFitness] = batAlgorithm(options);

% Display the best cluster centers
fprintf('Best cluster centers found: %s\n', mat2str(bestSol));
fprintf('Best fitness (sum of squared distances): %.4f\n', bestFitness);

% Assign each data point to the nearest cluster
distances = pdist2(dataPoints', bestSol(:)).^2;  % Compute squared distances
[~, closestCluster] = min(distances, [], 2);

% Plot the results
figure;
hold on;
colors = ['r', 'g', 'b'];
for i = 1:nClusters
    clusterPoints = dataPoints(closestCluster == i);
    plot(clusterPoints, zeros(size(clusterPoints)), [colors(i) 'o'], 'MarkerSize', 8, 'DisplayName', ['Cluster ' num2str(i)]);
    plot(bestSol(i), 0, [colors(i) 'x'], 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', ['Center ' num2str(i)]);
end
xlabel('Data Points');
legend('show');
hold off;
