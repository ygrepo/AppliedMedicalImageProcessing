function SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m)
    % U: N x c matrix of membership values for each data point in each cluster
    % dataPoints: N x D matrix of data points
    % clusterCenters: c x D matrix of cluster centers
    % m: Fuzziness exponent (usually > 1)
    
    % Get the number of data points (N) and clusters (c)
    [N, ~] = size(U);
    
    % Numerator calculation: intra-cluster distances
    % Expand dataPoints to N x 1 x D and clusterCenters to 1 x c x D for vectorized computation
    dataPoints_expanded = permute(dataPoints, [1, 3, 2]); % N x 1 x D
    clusterCenters_expanded = permute(clusterCenters, [3, 1, 2]); % 1 x c x D
    
    % Calculate the squared distances between data points and cluster centers
    distances = sum((dataPoints_expanded - clusterCenters_expanded).^2, 3); % N x c
    
    % Weight these distances using the membership values raised to the power m
    numerator = sum((U.^m) .* distances, 1); % 1 x c
    
    % Denominator calculation: inter-cluster distances
    % Calculate the squared distances between cluster centers
    cluster_distances = sum((permute(clusterCenters, [1, 3, 2]) - permute(clusterCenters, [3, 1, 2])).^2, 3); % c x c
    
    % Add a small epsilon to avoid division by zero
    epsilon = 1e-10;
    denominator = N * sum(cluster_distances, 2)' + epsilon; % 1 x c
    
    % Compute the partition index for each cluster and sum them
    SC = sum(numerator ./ denominator);
end
