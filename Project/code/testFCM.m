% Load an example MRI image
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Select the 15th slice of the MRI
slice_number = 15;
sliceData = D(:,:,slice_number);  % uint8 data

% Convert uint8 data to double and normalize to [0, 1]
data = double(sliceData(:)) / 255;  % Normalizing the pixel values

% Number of clusters (e.g., brain tissue types)
nClusters = 4;

opt = fcmOptions(NumClusters = nClusters);
disp('FCM Options')
disp(opt)

% Apply Fuzzy C-Means (FCM) clustering
[centers, U] = fcm(data, opt);

% Get the cluster membership for each pixel
[~, maxU] = max(U);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segImg = reshape(maxU, size(sliceData));

% ----- Fixed Colors for the Clusters -----
% Define a fixed colormap (you can adjust these colors as needed)
fixedColors = [
    1 0 0;   % Red for Cluster 1
    0 1 0;   % Green for Cluster 2
    0 0 1;   % Blue for Cluster 3
    1 1 0;   % Yellow for Cluster 4
];

% Map each cluster label to its corresponding RGB color
clusteredImg = label2rgb(segImg, fixedColors);  % Use the fixed color map

% Display the original and clustered MRI slice
figure;
subplot(1, 2, 1);
imshow(sliceData, []);  % Display the original MRI slice
title('Original MRI Slice');

subplot(1, 2, 2);
imshow(clusteredImg);  % Display the segmented MRI slice with fixed colors
title(['Segmented MRI Slice with ', num2str(nClusters), ' Clusters']);

% ----- Adding a Proper Legend with Cluster Colors -----

% Create legend entries for each cluster
legendEntries = cell(nClusters, 1);
for i = 1:nClusters
    legendEntries{i} = sprintf('Cluster %d', i);
end

% Create invisible plots for each cluster color (for the legend)
hold on;
for i = 1:nClusters
    plot(NaN, NaN, 's', 'MarkerSize', 10, 'MarkerFaceColor', fixedColors(i, :), ...
        'MarkerEdgeColor', fixedColors(i, :));
end

% Add the legend with the fixed colors for each cluster
legend(legendEntries, 'Location', 'bestoutside');
hold off;

% ----- Mark the Cluster Centers on the Segmented Image -----

% Convert centers from normalized [0, 1] back to the original intensity scale [0, 255]
scaledCenters = centers * 255;

% Find the pixels in the original slice that are closest to the centers
nRows = size(sliceData, 1);
nCols = size(sliceData, 2);
hold on;
for i = 1:nClusters
    % Find the pixel closest to each cluster center
    [~, idx] = min(abs(double(sliceData(:)) - scaledCenters(i)));  % Find closest pixel to the center
    [x, y] = ind2sub([nRows, nCols], idx);  % Convert the 1D index to 2D coordinates (row, col)
    
    % Plot the center as a black cross ('kx') on the segmented MRI image
    plot(y, x, 'kx', 'MarkerSize', 12, 'LineWidth', 2, 'HandleVisibility', 'off');  % Mark cluster center without adding to legend
end
hold off;
