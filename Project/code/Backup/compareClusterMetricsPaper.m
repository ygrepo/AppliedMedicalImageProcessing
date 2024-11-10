clc;
options = struct();
options.nBats = 50;
options.itermax = 100;
options.lowerBound = 0;
options.upperBound = 1;
options.nClusters = 3;
options.m = 2; % Fuzziness exponent
options.fmin = 0;
options.fmax = 2;
options.A = 0.9; % Initial loudness
options.loudnessCoefficient = .9;
options.r = 0.1; % Initial pulse rate
options.gamma = 0.9; % Decay rate for pulse rate
options.chaotic = false;

options.epsilon = 1e-5; % Convergence criterion for FCM
options.fcmIterMax = 100;
options.DistanceMetric = 'euclidean';

options.verbose = false;

% Load and segment an example image
image = imread('mri.tif'); % Replace with your MRI image path
nBins = 50;
K = histeq(image,nBins);
img = double(K); % Convert image to double
img = img / max(img(:)); % Normalize image
img = img(:);
options.dataPoints = img;

results = computeFCM(options);
[~, maxU] = max(results.U); 
segmented_image = reshape(maxU, size(K));
figure;
imshow(segmented_image, []);

PC = calculatePartitionCoefficient(results.U);
CE = calculateClassificationEntropy(results.U);
SC = calculatePartitionIndex(results.U', options.dataPoints, ...
    results.centers, options.m);
S = fuzzySeparationIndex(options.dataPoints, results.centers,...
    results.U, options.m);
sprintf("PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f", PC,CE,SC, S)
results = MFBAFCM(options);
[~, maxU] = max(results.U); 
segmented_image = reshape(maxU, size(K));
figure;
imshow(segmented_image, []);

results.info
PC = calculatePartitionCoefficient(results.U);
CE = calculateClassificationEntropy(results.U);
SC = calculatePartitionIndex(results.U', options.dataPoints, ...
    results.centers, options.m);
S = fuzzySeparationIndex(options.dataPoints, results.centers,...
    results.U, options.m);
sprintf("PC:%5.3f-CE:%5.3f-SC:%5.3f-S:%5.3f", PC,CE,SC, S)