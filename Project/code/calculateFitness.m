function fitness = calculateFitness(clusterCenters, options)
    % dataPoints: N×D (number of data points × number of features).
    % clusterCenters: C×D (number of clusters × number of features).

    if isfield(options, 'dataPoints')

        % Extract the data points and number of clusters from options
        dataPoints = options.dataPoints;
        m = options.m; % Fuzziness exponent
    
        % Calculate intra-cluster distance
        intra_cluster = calculateIntraCluster(dataPoints, clusterCenters); % No transpose needed
    
        % Compute distances between data points and cluster centers (N x C)
        distances = pdist2(dataPoints, clusterCenters).^2; % Squared distances
    
        % Avoid division by zero by setting very small values to a small epsilon
        epsilon = 1e-10;
        distances(distances < epsilon) = epsilon;
    
        % Update membership matrix U using vectorized operations
        exponent = 2 / (m - 1);
        invDistances = 1 ./ distances; % Element-wise inversion of distances
        sumInvDistances = sum(invDistances .^ exponent, 2); % Sum across clusters for each data point
    
        U = (invDistances .^ exponent) ./ sumInvDistances; % Calculate membership values
    
        % Calculate partition index (SC)
        SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m);
    
        % Calculate partition coefficient (PC)
        PC = calculatePartitionCoefficient(U);
    
        % Compute the fitness value
        fitness = (intra_cluster + SC) / PC;
    else
        fitness = options.fitness(clusterCenters); % Call the provided fitness function
    end
end

