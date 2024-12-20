function [bestSol, bestFitness] = batAlgorithm(options)
    % Initialize parameters
    pulseRates = options.pulseRate * ones(options.nBats, 1);  % Per-bat pulse rates
    loudnesses = options.loudness * ones(options.nBats, 1);  % Per-bat loudness

    % Initialize bat positions for cluster centers and velocities
    bats = options.lowerBound + (options.upperBound - options.lowerBound) .* rand(options.nBats, options.nClusters);
    velocities = zeros(options.nBats, options.nClusters);
    fitness = zeros(options.nBats, 1);

    % Evaluate initial fitness for all bats
    for i = 1:options.nBats
        fitness(i) = calculateFitness(bats(i, :)', options);  % Call to calculateFitness
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestSol = bats(idx, :);

    % Main loop for the BAT algorithm
    for t = 1:options.itermax
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            Q = options.Qmin + (options.Qmax - options.Qmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestSol) * Q;
            newSolution = bats(i, :) + velocities(i, :);

            % Enforce boundary constraints
            newSolution = enforceBoundaries(newSolution, options.lowerBound, options.upperBound);

            % Local search
            if rand > pulseRates(i)
                newSolution = bestSol + 0.01 * randn(1, options.nClusters);
            end

            % Evaluate the new solution's fitness
            newFitness = calculateFitness(newSolution', options);

            % Acceptance criteria
            if (newFitness < fitness(i)) && (rand < loudnesses(i))
                bats(i, :) = newSolution;
                fitness(i) = newFitness;
                loudnesses(i) = options.loudnessCoefficient * loudnesses(i);  % Decrease loudness
                pulseRates(i) = pulseRates(i) * (1 - exp(-options.gamma * t));  % Increase pulse rate
            end

            % Update global best
            if newFitness < bestFitness
                bestSol = newSolution;
                bestFitness = newFitness;
            end
        end
    end
end

% Function to enforce boundary constraints
function newSolution = enforceBoundaries(newSolution, lowerBound, upperBound)
    newSolution = max(min(newSolution, upperBound), lowerBound);
end
