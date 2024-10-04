function segmented_image = MFBAFCM(image, options)
    % Step 1: Input original image and set parameters
    img = double(image); % Convert image to double
    img = img / max(img(:)); % Normalize image
    options.dataPoints = img(:); % Flatten image for clustering
    options.dim = options.nClusters; % Number of clusters is the dimension

    % Step 2: Run the Modified Bat Algorithm (MBA) to find initial cluster centers
    [optimalCenters, ~] = batAlgorithm(options);

    % Step 3: Run FCM with the initial cluster centers found by MBA
    %[finalCenters, U] = customFCM(options.dataPoints, options.nClusters, options, optimalCenters);

    size(optimalCenters)
    opt = fcmOptions(MinImprovement = options.epsilon,...
    ClusterCenters = optimalCenters',...
    NumClusters='auto',...
    MaxNumIteration=options.fcmIterMax);
    [finalCenters, U] = fcm(options.dataPoints, opt);

    % Step 4: Reshape the membership matrix U to form the segmented image
    [~, maxU] = max(U); % Find the cluster with the highest membership for each pixel
    segmented_image = reshape(maxU, size(img));
end


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

    % Main loop for MBA
    for t = 1:options.itermax
        fprintf("Iter.:%d\n", t)
        for i = 1:options.nBats
            %fprintf("Bats.:%d\n", i)
            % Update frequency, velocity, and position
            Q = options.fmin + (options.fmax - options.fmin) * rand;
            velocities(i, :) = velocities(i, :) + (bats(i, :) - bestSol) * Q;
            newSolution = bats(i, :) + velocities(i, :);

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

function fitness = calculateFitness(clusterCenters, options)
    % Extract the data points and number of clusters from options
    dataPoints = options.dataPoints;
    nClusters = options.nClusters;
    m = options.m; % Fuzziness exponent

    % Number of data points
    N = length(dataPoints);

    % Calculate intra-cluster distance
    intra_cluster = calculateIntraCluster(dataPoints, clusterCenters);

    % Initialize the membership matrix (U) based on the current cluster centers
    U = zeros(nClusters, N);

    % Compute distances between data points and cluster centers
    distances = pdist2(dataPoints, clusterCenters).^2; % Squared distances

    % Update membership matrix U based on current cluster centers
    for j = 1:nClusters
        for i = 1:N
            % Calculate membership value for each cluster
            if distances(i, j) == 0
                % Avoid division by zero; assign full membership
                U(:, i) = 0;
                U(j, i) = 1;
            else
                U(j, i) = 1 / sum((distances(i, j) ./ distances(i, :)).^(2 / (m - 1)));
            end
        end
    end
    U = U';

    % Calculate partition index (SC)
    SC = calculatePartitionIndex(U, dataPoints, clusterCenters, m);
    
    % Calculate partition coefficient (PC)
    PC = calculatePartitionCoefficient(U);

    % Compute the fitness value
    fitness = (intra_cluster + SC) / PC;%
    % fprintf("fitness:%5.2f\n", fitness);
end


function [clusterCenters, U] = customFCM(dataPoints, nClusters, options, initialCenters)
    % Get the number of data points
    N = length(dataPoints);
    m = options.m; % Fuzziness exponent
    itermax = options.fcmIterMax; % Maximum number of iterations
    epsilon = options.epsilon; % Convergence criterion

    % Initialize cluster centers
    clusterCenters = initialCenters;

    % Initialize the membership matrix (U)
    U = rand(nClusters, N);
    U = U ./ sum(U, 1); % Normalize to ensure the sum of memberships for each data point is 1

    % Iterate until convergence or max iterations
    for iter = 1:itermax
        U_prev = U; % Store the previous membership matrix

        % Update cluster centers
        for j = 1:nClusters
            numerator = sum((U(j, :) .^ m) .* dataPoints');
            denominator = sum(U(j, :) .^ m);
            clusterCenters(j) = numerator / denominator;
        end

        % Update membership matrix U
        for i = 1:N
            for j = 1:nClusters
                dist = abs(dataPoints(i) - clusterCenters(j));
                if dist == 0
                    U(:, i) = 0;
                    U(j, i) = 1;
                else
                    U(j, i) = 1 / sum((dist / abs(dataPoints(i) - clusterCenters(:))).^(2 / (m - 1)));
                end
            end
        end

        % Check for convergence
        if max(max(abs(U - U_prev))) < epsilon
            break;
        end
    end
end
