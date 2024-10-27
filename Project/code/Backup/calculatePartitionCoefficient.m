function PC = calculatePartitionCoefficient(U)
    % U: N x c matrix of membership values for each data point in each cluster
    
    % Get the number of data points (N)
    N = size(U, 1);
    
    % Calculate the partition coefficient
    PC = sum(U.^2, 'all') / N;
end
