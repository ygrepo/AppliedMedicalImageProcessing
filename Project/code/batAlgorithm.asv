
function [bestSol, bestFitness] = batAlgorithm(options)
    % Initialize parameters
    r = options.r;
    A = options.A;

    % Initialize bat positions for cluster centers and velocities
    bats = options.lowerBound + (options.upperBound - options.lowerBound) * rand(options.nBats, options.dim);
    velocities = zeros(options.nBats, options.dim);
    fitness = zeros(options.nBats, 1);

    % Step 2: Evaluate initial fitness for all bats
    for i = 1:options.nBats
        fitness(i) = calculateFitness(bats(i, :)', options); % Call to calculateFitness
    end

    % Find the initial best solution
    [bestFitness, idx] = min(fitness);
    bestSol = bats(idx, :);

    % Chaotic map (sinusoidal as an example)
    chaotic_map = @(x) sin(2 * pi * x);

    % Main loop for MBA
    for t = 1:options.itermax
        fprintf("Iter.:%d\n", t)
        chaotic_value = chaotic_map(mod(t, options.itermax) / options.itermax);
        for i = 1:options.nBats
            % Update frequency, velocity, and position
            Q = options.fmin + (options.fmax - options.fmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestSol) * Q;
            newSolution = bats(i, :) + velocities(i, :);
            if options.chaotic
                newSolution = newSolution + chaotic_value * (chaotic_map(chaotic_value) - 0.5);
            end
            % Enforce boundary constraints
            newSolution = max(min(newSolution, options.upperBound), options.lowerBound);

            % Local search
            if rand > r
                newSolution = bestSol + 0.01 * randn(1, options.dim);
            end
 
            % Step 3: Evaluate the new solution's fitness
            newFitness = calculateFitness(newSolution', options); % Call to calculateFitness

            % Acceptance criteria
            if (newFitness < fitness(i)) && (rand < A)
                bats(i, :) = newSolution;
                fitness(i) = newFitness;
                A = options.loudnessCoefficient * A; % Decrease loudness
                r = r * (1 - exp(-options.gamma * t)); % Increase pulse rate
            end

            % Replace unchanged bats
            if sum(fitness == fitness(i)) >= 4
                [~, sortedIdx] = sort(fitness);
                bestFive = mean(bats(sortedIdx(1:5), :));
                bats(i, :) = bestFive;
            end

            % Update global best
            if newFitness < bestFitness
                bestSol = newSolution;
                bestFitness = newFitness;
            end
        end
    end
end
