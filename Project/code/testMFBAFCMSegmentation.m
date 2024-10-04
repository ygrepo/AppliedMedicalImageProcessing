% Example parameters
options.nBats = 30;
options.itermax = 100;
options.lowerBound = 0;
options.upperBound = 1;
options.nClusters = 4;
options.m = 2; % Fuzziness exponent
options.fmin = 0;
options.fmax = 2;
options.A = 0.5; % Initial loudness
options.loudnessCoefficient = .9;
options.r = 0.5; % Initial pulse rate
options.gamma = 0.9; % Decay rate for pulse rate


options.epsilon = 1e-5; % Convergence criterion for FCM
options.fcmIterMax = 200;

% Load and segment an example image
img = imread('mri.tif'); % Replace with your MRI image path
%Display the original image
figure;
imshow(img, []);
title('Original MRI Image');
hold off
segmented_image = MFBAFCM(img, options);
    
% Step 5: Output segmented image
figure;
imshow(segmented_image, []);
title('Segmented MRI Image');