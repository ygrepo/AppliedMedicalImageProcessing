% Example parameters
options = struct();
options.nBats = 30;
options.itermax = 100;
options.lowerBound = 0;
options.upperBound = 1;
options.nClusters = 4;
options.dim = options.nClusters; % Number of clusters is the dimension
options.m = 2; % Fuzziness exponent
options.fmin = 0;
options.fmax = 2;
options.A = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.r = 0.5; % Initial pulse rate
options.gamma = 0.9; % Decay rate for pulse rate
options.chaotic = false;

options.epsilon = 1e-5; % Convergence criterion for FCM
options.fcmIterMax = 200;
options.DistanceMetric = 'euclidean';

% Load and segment an example image
image = imread('mri.tif'); % Replace with your MRI image path
%Display the original image
figure;
imshow(image, []);
%title('Original MRI Image');
hold off

img = double(image); % Convert image to double
img = img / max(img(:)); % Normalize image
img = img(:);
options.dataPoints = img;

results = MFBAFCM(options);
% eshape the membership matrix U to form the segmented image
[~, maxU] = max(results.U); % Find the cluster with the highest membership for each pixel
segmented_image = reshape(maxU, size(image));

figure;
imshow(segmented_image, []);
%title('Segmented MRI Image');

options.chaotic = true;
options.DistanceMetric = 'fmle';
results = MFBAFCM(options);
[~, maxU] = max(results.U); % Find the cluster with the highest membership for each pixel
segmented_image = reshape(maxU, size(image));
figure;
imshow(segmented_image, []);
