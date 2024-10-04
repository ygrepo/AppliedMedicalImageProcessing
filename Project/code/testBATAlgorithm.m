
% Objective function: A simple quadratic function generalized for multi-dimensions
function f = fitness(x)
    % If x is a vector, the sum of squares computes the generalized Sphere function
    f = sum((x - 2).^2);
end

% Example usage for 1D problem
options = struct();
options.nBats = 20; % Number of bats in the population
options.nIterations = 100; % Number of iterations
options.lowerBound = -10; % Lower bound of the search space
options.upperBound = 10; % Upper bound of the search space
options.dim = 1; % Dimensionality of the problem (1D)
options.A = 0.5; % Initial loudness
options.r = 0.5; % Initial pulse rate
options.Qmin = 0; % Minimum frequency
options.Qmax = 2; % Maximum frequency
options.gamma = 0.1;
options.fitness = @fitness;

[bestSol, bestFitness] = batAlgorithm(options);
fprintf('Best solution (1D) found: %s\n', mat2str(bestSol));
fprintf('Best fitness (1D) found: %.4f\n', bestFitness);

% Example usage for multi-dimensional problem (e.g., 5D)
options.dim = 5; % Dimensionality of the problem (multi-dimensional)
[bestSol, bestFitness] = batAlgorithm(options);
fprintf('Best solution (multi-dimensional) found: %s\n', mat2str(bestSol));
fprintf('Best fitness (multi-dimensional) found: %.4f\n', bestFitness);
