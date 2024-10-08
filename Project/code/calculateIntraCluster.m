function intra_cluster = calculateIntraCluster(dataPoints, clusterCenters)
    % dataPoints: N x D matrix (N data points, D features)
    % clusterCenters: c x D matrix (c cluster centers, D features)
    
    % Get the number of data points and cluster centers
    [N, D] = size(dataPoints); % N: number of data points, D: features
    c = size(clusterCenters, 1); % c: number of cluster centers
    
    % Expand dataPoints and clusterCenters for vectorized computation
    dataPoints_expanded = permute(dataPoints, [1, 3, 2]); % N x 1 x D
    clusterCenters_expanded = permute(clusterCenters, [3, 1, 2]); % 1 x c x D
    
    % Calculate the squared distances between each data point and each cluster center
    distances = sum((dataPoints_expanded - clusterCenters_expanded).^2, 3); % N x c matrix
    
    % Sum all distances and average by the number of data points
    intra_cluster = sum(distances, 'all') / N;
end