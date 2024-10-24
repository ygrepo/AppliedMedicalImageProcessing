clearvars 
vol = niftiread('sub-11_T1w.nii');
%%
sliceViewer(vol);
% montage(vol)
% title('Original image volume')
%volumeViewer(V)
%% Generate noisy images ----
 volsigma10 = applyNoise(vol, 10);
 volsigma20 = applyNoise(vol, 20);
 volsigma30 = applyNoise(vol, 30);
%%
%sliceViewer(volsigma10);
sliceNumber = 102;
% Display slice 102 using imshow
figure;
imshow(vol(:,:,sliceNumber), []);
title('Slice 102');
hold off;

% Display slice 102 using imshow
figure;
imshow(volsigma10(:,:,sliceNumber), []);
title('Slice 102');
hold off;
%% Gaussian smoothing Filter ----
siz= size(vol);
sigma = .1;
volSmooth = imgaussfilt3(volsigma10, sigma);
sliceNumber = 102;
volsigma10Slice102 = volsigma10(:,:,sliceNumber);
volsigma10SmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigma10Slice119 = volsigma10(:,:,sliceNumber);
volsigma10SmoothedSlice119 = volSmooth(:,:,sliceNumber);

%%
figure;
t = tiledlayout(1, 2);
%title(t, titleText);
nexttile;  
imshow(volsigma10Slice102, []);
title('Slice 102 Noisy Image');

nexttile;  
imshow(volsigma10SmoothedSlice102, []);
title('Slice 102 Gaussian Filter on Noisy Image');
%%
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigma10Slice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigma10SmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigma10Slice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigma10SmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);


figure;

% Display original and smoothed slice 102
subplot(2, 2, 1);
imshow(roi_102, []);  % Display the original ROI from slice 102
title('Original Slice 102 - ROI');

subplot(2, 2, 2);
imshow(roi_102_smoothed, []);  % Display the smoothed ROI from slice 102
title('Smoothed Slice 102 - ROI');

% % Display original and smoothed slice 119
subplot(2, 2, 3);
imshow(roi_119, []);  % Display the original ROI from slice 119
title('Original Slice 119 - ROI');

subplot(2, 2, 4);
imshow(roi_119_smoothed, []);  % Display the smoothed ROI from slice 119
title('Smoothed Slice 119 - ROI');



%%
sliceNumber = 102;
montage({volsigma10(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
title("Original Slice (Left) Vs. Gaussian Filtered Slice (Right)")

%%
slice102 = vol(:, :, 102);  % Extract slice 102
slice119 = vol(:, :, 119);  % Extract slice 119


% Display slice 102 using imshow
figure;
imshow(vol(:,:,sliceNumber), []);
title('Slice 102');
hold off;
%%
% Display slice 119 using imshow
figure;
imshow(slice119, []);
title('Slice 119');



%%
[~, ~, dim3] = size(vol);  % Get dimensions of the 3D image
sigma = .1;  % Define the standard deviation for Gaussian filter
volSmooth = zeros(size(vol));  % Initialize an array to store the filtered image

for i = 1:dim3
    slice = vol(:, :, i);  % Extract the ith slice
    filteredSlice = imgaussfilt(slice, sigma);  % Apply Gaussian filter
    volSmooth(:, :, i) = filteredSlice;  % Store the filtered slice back
end

figure
sliceNumber = 102;
montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
title("Original Slice (Left) Vs. Gaussian Filtered Slice (Right)")


%% Median Filter ----
volSmooth = medfilt3(vol);
sliceNumber = 102;
montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
title("Original Slice (Left) Vs. Median Filter Slice (Right)")

%%
windowSize = 3;  % Define the window size for the median filter
[dim1, dim2, dim3] = size(vol); 
volSmooth = zeros(size(vol));  % Initialize an array to store the filtered image

for i = 1:dim3
    slice = vol(:, :, i);  % Extract the ith slice
    %filteredSlice = medfilt2(slice);  % Apply median filter
    filteredSlice = medfilt2(slice, [windowSize windowSize]);  % Apply median filter
    volSmooth(:, :, i) = filteredSlice;  % Store the filtered slice back
end
sliceNumber = 15;
montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
title("Original Slice (Left) Vs. Median Filter Slice (Right)")

function Innoisy = applyNoise(I, s)
I = flipdim (permute(I, [2 1 3]), 1);
I = double (I);
dim = size(I);
x = s .* randn(dim) + I;
y = s .* randn(dim);
Innoisy = sqrt(x.^2+y.^2);
end