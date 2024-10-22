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
