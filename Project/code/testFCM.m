% Load an example MRI image
img = imread('mri.tif'); % Load an MRI image (adjust path if necessary)
img = double(img); % Convert image to double for processing

% Display the original image
figure;
imshow(img, []);
title('Original MRI Image');

% Normalize the image intensity
img = img / max(img(:));

% Number of clusters (e.g., brain tissue types)
nClusters = 4;

% Apply Fuzzy C-Means (FCM) clustering
% Reshape the image to a 1D array for clustering
data = img(:);
[centers, U] = fcm(data, nClusters);

% Get the cluster membership for each pixel
[~, maxU] = max(U); % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segmented_img = reshape(maxU, size(img));

% Display the segmented image
figure;
imshow(segmented_img, []);
title('Segmented MRI Image with FCM');

% Optionally, display each cluster separately
% for i = 1:nClusters
%     cluster = reshape(U(i, :), size(img)); % Get the membership of the i-th cluster
%     figure;
%     imshow(cluster, []);
%     title(['Membership of Cluster ', num2str(i)]);
% end
