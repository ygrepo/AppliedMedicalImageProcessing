function intraCluster = calculateIntraCluster(dataPoints, clusterCenters)
    % dataPoints: N x D matrix (N data points, D features)
    % clusterCenters: c x D matrix (c cluster centers, D features)
    
    % Compute the pairwise squared Euclidean distances between data points and cluster centers
    distances = pdist2(dataPoints, clusterCenters, 'squaredeuclidean');
    
    % Sum all distances and average by the number of data points
    intraCluster = sum(distances, 'all') / size(dataPoints, 1);
end