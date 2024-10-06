function S = fuzzySeparationIndex(data, centroids, U, m)
    % data: matrix of size [num_samples, num_features]
    % centroids: matrix of size [num_clusters, num_features]
    % U: N x c matrix of membership values for each data point in each cluster
    % m: fuzziness exponent (typically m > 1)
    
    % Number of samples and clusters
    [num_samples, ~] = size(data);
    [num_clusters, ~] = size(centroids);
    
    % Calculate the numerator
    numerator = 0;
    for j = 1:num_clusters
        for i = 1:num_samples
            % Membership value raised to power m
            u_ji_m = U(i, j)^m;
            % Squared Euclidean distance between data point and cluster centroid
            dist_ij = norm(data(i, :) - centroids(j, :))^2;
            numerator = numerator + u_ji_m * dist_ij;
        end
    end
    
    % Calculate the denominator
    min_inter_centroid_dist = inf;
    for j = 1:num_clusters
        for k = j+1:num_clusters
            % Squared Euclidean distance between centroids
            dist_jk = norm(centroids(j, :) - centroids(k, :))^2;
            if dist_jk < min_inter_centroid_dist
                min_inter_centroid_dist = dist_jk;
            end
        end
    end
    
    % Calculate the separation index
    denominator = num_samples * min_inter_centroid_dist;
    S = numerator / denominator;
end
