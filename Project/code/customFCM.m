

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
