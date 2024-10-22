% Load an example MRI image
clearvars 
load mri;  % Preloaded data from MATLAB
D = squeeze(D);  % Removes singleton dimensions

% Select the 15th slice of the MRI
slice_number = 15;
sliceData = D(:,:,slice_number);  % uint8 data
sliceData = histeq(sliceData);

% Convert uint8 data to double and normalize to [0, 1]
data = double(sliceData(:)) / 255;  % Normalizing the pixel values

nClusters = 4;

opt = fcmOptions(NumClusters = nClusters);
disp('FCM Options')
disp(opt)

% Apply Fuzzy C-Means (FCM) clustering
[fcmCenters, U] = fcm(data, opt);

% Get the cluster membership for each pixel
[~, maxU] = max(U);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segFCMImg = reshape(maxU, size(sliceData));



% Example parameters
options = struct();
options.nBats = 30;
options.itermax = 100;
options.lowerBound = 0;
options.upperBound = 1;
options.nClusters = 4;
options.m = 2; % Fuzziness exponent
options.Qmin = 0;
options.Qmax = 2;
options.loudness = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.pulseRate = 0.5; % Initial pulse rate
options.gamma = 0.9; % Decay rate for pulse rate
options.chaotic = false;

options.epsilon = 1e-5; % Convergence criterion for FCM
options.fcmIterMax = 200;
options.DistanceMetric = 'euclidean';
options.dataPoints = data;


% Apply BAT + Fuzzy C-Means (FCM) clustering
segImgInresults = MFBAFCM(options);

% Get the cluster membership for each pixel
[~, maxU] = max(segImgInresults.U);  % Determine the cluster with the highest membership for each pixel

% Reshape the result back to the original image size
segBATFCMImg = reshape(maxU, size(sliceData));

% ----- Fixed Colors for the Clusters -----
fixedColors = [
    1 0 0;   % Red for Cluster 1
    0 1 0;   % Green for Cluster 2
    0 0 1;   % Blue for Cluster 3
    1 1 0;   % Yellow for Cluster 4
];

% Display the original and clustered MRI slice
figure;
h1= subplot(1, 3, 1);
imshow(sliceData, []);  % Display the original MRI slice
title('Original MRI Slice', 'FontSize', 20, 'FontWeight','bold');

h2 = subplot(1, 3, 2);
opt = struct();
opt.title = ['Segmented MRI Slice with FCM ', num2str(nClusters), ' Clusters'];
centers = [fcmCenters'];
opt.centerColors = ['kx'];
opt.fixedColors = fixedColors; 
opt.centerNames = ['FCM Centers'];
showSegmentedImg(sliceData, segFCMImg, centers, opt);


h3 = subplot(1, 3, 3);
opt = struct();
opt.title = ['Segmented MRI Slice with BAT + FCM ', num2str(nClusters), ' Clusters'];
centers = [segImgInresults.batCenters];
opt.centerColors = ['rx'];
opt.fixedColors = fixedColors; 
opt.centerNames = ['Bat Centers'];
showSegmentedImg(sliceData, segBATFCMImg, centers, opt);

% 'Position' is [left, bottom, width, height]
set(h1, 'Position', [0.05, 0.1, 0.25, 0.8]);  % Adjust as needed
set(h2, 'Position', [0.35, 0.1, 0.25, 0.8]);  % Adjust as needed
set(h3, 'Position', [0.65, 0.1, 0.25, 0.8]);  % Adjust as needed