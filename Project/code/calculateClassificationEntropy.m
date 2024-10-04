function CE = calculateClassificationEntropy(U)
    % U: N x c matrix of membership values for each data point in each cluster
    
    % Get the number of data points (N)
    N = size(U, 1);
    
    % Avoid log(0) by adding a very small value (epsilon) to U
    epsilon = 1e-10;
    
    % Compute the classification entropy
    CE = -sum(U .* log(U + epsilon), 'all') / N;
end
