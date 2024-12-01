function showCenters(Img, centers, opt)
    % Show the original image
    imshow(Img, []);
    hold on;

    % Set default values for optional parameters if not provided
    if ~isfield(opt, 'nClusters')
        opt.nClusters = size(centers, 1);  % Set based on the number of centers
    end

    % Ensure fixedColors has enough colors for all clusters
    if ~isfield(opt, 'centerColors') || size(opt.centerColors, 1) < opt.nClusters
        % Generate a colormap if insufficient colors are provided
        opt.centerColors = lines(opt.nClusters);  % Default to distinct colors
    end

    % Ensure center names are provided for each cluster
    if ~isfield(opt, 'centerNames')
        opt.centerNames = arrayfun(@(x) sprintf('Cluster %d', x), 1:opt.nClusters, 'UniformOutput', false);
    end

    % Extract 3x3 neighborhoods from the image as feature vectors
    [nRows, nCols] = size(Img);
    patchDim = sqrt(size(centers, 2));  % Assuming a 3x3 window for each center (9 elements)
    halfPatch = floor(patchDim / 2);

    % Pad the image to extract neighborhoods at the borders
    paddedImg = padarray(Img, [halfPatch, halfPatch], 'symmetric');
    featureVectors = im2col(paddedImg, [patchDim, patchDim], 'sliding')';

    % Match each center to the closest neighborhood in the image
    for i = 1:opt.nClusters
        % Get the i-th center feature vector
        centerVector = centers(i, :);

        % Compute Euclidean distances from center to all neighborhoods
        dists = sqrt(sum((featureVectors - centerVector).^2, 2));

        % Find the index of the closest neighborhood
        [~, closestIdx] = min(dists);

        % Convert linear index back to row, col in the original image
        [row, col] = ind2sub([nRows, nCols], closestIdx);

        % Plot the center with specified color and label
        plot(col, row, 'x', 'Color', opt.centerColors(i, :), 'MarkerSize', 18, 'LineWidth', 4, ...
            'DisplayName', opt.centerNames{i});
    end

    % Show legend with center names
    legend('show', 'Location', 'bestoutside');
    hold off;
end
