function S = fuzzySeparationIndex(data, centroids, U, m)
    % data: matrix of size [num_samples, num_features]
    % centroids: matrix of size [num_clusters, num_features]
    % U: N x c matrix of membership values for each data point in each cluster
    % m: fuzziness exponent (typically m > 1)
    
    % Number of samples and clusters
    [num_samples, num_features] = size(data);
    [num_clusters, ~] = size(centroids);
    
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
