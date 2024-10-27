% Sample data
dataPoints = [1.0, 2.0; 3.0, 4.0; 5.0, 6.0]; % 3 data points, 2 features
clusterCenters = [2.0, 3.0; 4.0, 5.0]; % 2 clusters, 2 features

% Calculate intra-cluster variance using the vectorized function
intra_cluster = calculateIntraCluster(dataPoints, clusterCenters);

% Display the result
fprintf('Intra-cluster variance (vectorized): %.2f\n', intra_cluster);
%%

% Sample membership matrix U (N = 3 data points, c = 2 clusters)
U = [1, 1; 1, 1; 1, 1];

% Calculate the partition coefficient
PC = calculatePartitionCoefficient(U);

% Display the result
fprintf('Partition Coefficient (PC): %.4f\n', PC);
%%
% Sample membership matrix U (N = 3 data points, c = 2 clusters)
U = [0.8, 0.2; 0.4, 0.6; 0.3, 0.7];

% Calculate the classification entropy
CE = calculateClassificationEntropy(U);

% Display the result
fprintf('Classification Entropy (CE): %.4f\n', CE);
%%
% Sample data
dataPoints = [1.0, 2.0; 3.0, 4.0; 5.0, 6.0]; % 3 data points, 2 features
clusterCenters = [2.0, 3.0; 4.0, 5.0]; % 2 clusters, 2 features
U = [0.8, 0.2; 0.4, 0.6; 0.3, 0.7]; % Membership matrix
m = 2; % Fuzziness exponent

% Calculate the partition index
SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m);

% Display the result
fprintf('Partition Index (SC): %.4f\n', SC);

