
clearvars
clc
vol = niftiread('data/sub-11_T1w.nii.gz');
vol = flip (permute(vol, [2 1 3]), 1);

% bestParams = gridSearch(vol);

% Look at Slice 102 and Slice 119 ---
sliceNumber = 102;
% Display slice 102 using imshow
figure;
imshow(vol(:,:,sliceNumber), []);
title('Slice 102');
hold off;

% Display slice 102 using imshow
figure;
sliceNumber = 119;
imshow(vol(:,:,sliceNumber), []);
title('Slice 119');
hold off;

% Generate noisy images ----
noisyImages = {};
noisyImages{1} = applyNoise(vol, 10);
noisyImages{2} = applyNoise(vol, 20);
noisyImages{3} = applyNoise(vol, 30);
showNoisyImages(noisyImages);

% Gaussian filtering of noisy images ----
gaussianFilterSigma = .5;
gaussianDenoisedVols = {};
gaussianDenoisedVols{1} = imgaussfilt3(noisyImages{1}, gaussianFilterSigma);
gaussianDenoisedVols{2} = imgaussfilt3(noisyImages{2}, gaussianFilterSigma);
gaussianDenoisedVols{3} = imgaussfilt3(noisyImages{3}, gaussianFilterSigma);

% Median Filter of noisy images ----
medianFilterDenoisedVols = {};
medianFilterDenoisedVols{1} = medfilt3(noisyImages{1});
medianFilterDenoisedVols{2} = medfilt3(noisyImages{2});
medianFilterDenoisedVols{3} = medfilt3(noisyImages{3});

% BilateralFilterGray of noisy images ----
sigmaD = 1;  % Standard deviation for the domain (spatial) Gaussian kernel
sigmaR = 30; % Standard deviation for the range Gaussian kernel
bilateralFilterDenoisedVols = {};
bilateralFilterDenoisedVols{1} =computeBilateralFilter(noisyImages{1}, sigmaD, sigmaR);
bilateralFilterDenoisedVols{2} =computeBilateralFilter(noisyImages{2}, sigmaD, sigmaR);
bilateralFilterDenoisedVols{3} =computeBilateralFilter(noisyImages{3}, sigmaD, sigmaR);

% Perona-Malik Filter of noisy images ----
alpha = 0.1;   % Update rate
kappa = 15;     % Smoothness parameter
T = 10;         % Number of iterations
peronaMalikFilterDenoisedVols = {};
peronaMalikFilterDenoisedVols{1} =computePeronaMalikFilter(noisyImages{1}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{2} =computePeronaMalikFilter(noisyImages{2}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{3} =computePeronaMalikFilter(noisyImages{3}, alpha, kappa, T);

% Define the region of interest (ROI) ----
xStart = [80, 110];
yStart = [50, 80];
width = [50, 50];
height = [50, 50];
sliceNumbers = [102, 119];
noisyRois = getROI(noisyImages, sliceNumbers, xStart, yStart, width, height);
gaussianDenoisedRois = getROI(gaussianDenoisedVols,... 
    sliceNumbers, xStart, yStart, width, height);
medianFilterDenoisedRois = getROI(medianFilterDenoisedVols,...
    sliceNumbers, xStart, yStart, width, height);
bilateralFilterDenoisedRois = getROI(bilateralFilterDenoisedVols,...
    sliceNumbers, xStart, yStart, width, height);
peronaMalikFilterDenoisedRois = getROI(peronaMalikFilterDenoisedVols,...
    sliceNumbers, xStart, yStart, width, height);

% Check ROIS of first noise level images (10) ----
clc
figure
I = noisyRois{1};
imshowpair(I{1},I{2},'montage')
title('Noisy ROI, 102, 119')
% Before and After Filtering, Noise scale = 10 ----
noiseIndex = 1;
scaleValue = 10;
medianSupportSize = "(3x3x3)";
medianPadopt = "Symmetric";
fontSize = 14;
titleText = sprintf('Noise, Scale=%d, Before and After Filtering for ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    gaussianFilterSigma,...
    gaussianDenoisedRois, ...
    medianSupportSize,...
    medianPadopt,...
    medianFilterDenoisedRois, ...
    sigmaD,...
    sigmaR,...
    bilateralFilterDenoisedRois, ...
    alpha,...
    kappa,...
    T,...
    peronaMalikFilterDenoisedRois,...
    fontSize);

% Before and After Filtering, Noise scale = 20 ----
noiseIndex = 2;
scaleValue = 20;
medianSupportSize = "(3x3x3)";
medianPadopt = "Symmetric";
fontSize = 14;
titleText = sprintf('Noise, Scale=%d, Before and After Filtering for ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    gaussianFilterSigma,...
    gaussianDenoisedRois, ...
    medianSupportSize,...
    medianPadopt,...
    medianFilterDenoisedRois, ...
    sigmaD,...
    sigmaR,...
    bilateralFilterDenoisedRois, ...
    alpha,...
    kappa,...
    T,...
    peronaMalikFilterDenoisedRois,...
    fontSize);
% Before and After Filtering, Noise scale = 30 ----
noiseIndex = 3;
scaleValue = 30;
medianSupportSize = "(3x3x3)";
medianPadopt = "Symmetric";
fontSize = 14;
titleText = sprintf('Noise, Scale=%d, Before and After Filtering for ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    gaussianFilterSigma,...
    gaussianDenoisedRois, ...
    medianSupportSize,...
    medianPadopt,...
    medianFilterDenoisedRois, ...
    sigmaD,...
    sigmaR,...
    bilateralFilterDenoisedRois, ...
    alpha,...
    kappa,...
    T,...
    peronaMalikFilterDenoisedRois,...
    fontSize);
% Voxel Squared Error and Plot ----
eValues = getFilterPerformance(vol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols);
% Noise Level 10, Voxel Squared Error - Slices (102,119) ---
titleText = sprintf('Noise Level %d, Voxel Squared Error - Slices (102,119)', 10);
plotFilterPerformances(titleText, 1, eValues);
% Noise Level 20, Voxel Squared Error - Slices (102,119) ---
titleText = sprintf('Noise Level %d, Voxel Squared Error - Slices (102,119)', 20);
plotFilterPerformances(titleText, 2, eValues);
% Noise Level 30, Voxel Squared Error - Slices (102,119) ---
titleText = sprintf('Noise Level %d, Voxel Squared Error - Slices (102,119)', 30);
plotFilterPerformances(titleText, 3, eValues);
% MSE and Plot ----
mseValues = getRMSE(vol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols);
plotMSE(mseValues);

%% applyNoise ----
function Innoisy = applyNoise(I, s)
% Generate a noisy image using Rician distribution
I = double (I);
dim = size(I);
x = s .* randn(dim) + I;
y = s .* randn(dim);
Innoisy = sqrt(x.^2+y.^2);
end

%% BilateralFilterGray ----
function filteredI = BilateralFilterGraySep(I, sigmaD, sigmaR)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigmaD  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigmaR  - standard deviation for the range Gaussian kernel
    % Output:
    %   filteredI - output filtered grayscale image of size M x N

    % Convert image to double for precision
    I = double(I);

    % Get the size of the image
    [M, N] = size(I);

    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigmaD);

    % First pass: Vertical filtering
    IPrime = zeros(M, N); % This will store the intermediate result

    for u = 1:M
        for v = 1:N
            % Initialize S and W
            S = 0;
            W = 0;

            % Get the intensity of the center pixel in the original image
            a = I(u, v);

            % Loop over the vertical neighborhood
            for m = -K:K
                % Ensure the neighboring pixel index is within bounds
                if (u + m >= 1) && (u + m <= M)
                    % Get the neighboring pixel (vertical axis) from the original image
                    b = I(u + m, v);

                    % Compute spatial (domain) Gaussian weight
                    wd = exp(-m^2 / (2 * sigmaD^2));

                    % Compute range Gaussian weight
                    wr = exp(-(a - b)^2 / (2 * sigmaR^2));

                    % Total weight
                    w = wd * wr;

                    % Update S and W
                    S = S + w * b;
                    W = W + w;
                end
            end
            % Set the intermediate filtered value for the vertical pass
            IPrime(u, v) = S / W;
        end
    end

    % Second pass: Horizontal filtering
    filteredI = zeros(M, N); % This will store the final result

    for u = 1:M
        for v = 1:N
            % Initialize S and W
            S = 0;
            W = 0;

            % Get the intensity of the center pixel from the intermediate result
            a = IPrime(u, v);

            % Loop over the horizontal neighborhood
            for n = -K:K
                % Ensure the neighboring pixel index is within bounds
                if (v + n >= 1) && (v + n <= N)
                    % Get the neighboring pixel (horizontal axis) from the intermediate result
                    b = IPrime(u, v + n);

                    % Compute spatial (domain) Gaussian weight
                    wd = exp(-n^2 / (2 * sigmaD^2));

                    % Compute range Gaussian weight
                    wr = exp(-(a - b)^2 / (2 * sigmaR^2));

                    % Total weight
                    w = wd * wr;

                    % Update S and W
                    S = S + w * b;
                    W = W + w;
                end
            end

            % Set the final filtered value for the horizontal pass
            filteredI(u, v) = S / W;
        end
    end
end

%% Perona-Malik Filter ----
function I = PeronaMalik(I, alpha, kappa, T)
    % Input:
    %   I      - input grayscale image of size M x N
    %   alpha  - update rate
    %   kappa  - smoothness parameter
    %   T      - number of iterations
    % Output:
    %   I      - output smoothed image

    % Get the size of the image
    [M, N] = size(I);

    % Define the conductivity function g(d)
    g = @(d) exp(-(d / kappa).^2);

    % Iterate over T iterations
    for t = 1:T
        % Initialize gradient maps Dx and Dy
        Dx = zeros(M, N);
        Dy = zeros(M, N);
        
        % Compute forward differences Dx and Dy
        for u = 1:M
            for v = 1:N
                if u < M
                    Dx(u, v) = I(u + 1, v) - I(u, v);
                else
                    Dx(u, v) = 0; % Zero padding at boundary
                end
                
                if v < N
                    Dy(u, v) = I(u, v + 1) - I(u, v);
                else
                    Dy(u, v) = 0; % Zero padding at boundary
                end
            end
        end

        % Update image based on the gradients and conductivity function
        for u = 1:M
            for v = 1:N
                % Get forward differences
                delta_0 = Dx(u, v);
                delta_1 = Dy(u, v);
                
                % Get backward differences
                if u > 1
                    delta_2 = -Dx(u - 1, v);
                else
                    delta_2 = 0; % Zero padding at boundary
                end
                
                if v > 1
                    delta_3 = -Dy(u, v - 1);
                else
                    delta_3 = 0; % Zero padding at boundary
                end
                
                % Update the pixel value using the Perona-Malik equation
                I(u, v) = I(u, v) + alpha * (...
                    g(abs(delta_0)) * delta_0 + ...
                    g(abs(delta_1)) * delta_1 + ...
                    g(abs(delta_2)) * delta_2 + ...
                    g(abs(delta_3)) * delta_3 );
            end
        end
    end
   
end


%% computeBilateralFilter ----
function filteredI = computeBilateralFilter(I, sigmaD, sigmaR)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    slice = I(:, :, i);  % Extract the ith slice
    slice = BilateralFilterGraySep(slice, sigmaD, sigmaR);  % Apply  filter
    filteredI(:, :, i) = slice;  % Store the filtered slice back
 end
end

%% computePeronaMalikFilter ----
function filteredI =computePeronaMalikFilter(I, alpha, kappa, T)
filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 

for i = 1:dim3
    slice = I(:, :, i);  % Extract the ith slice
    slice = PeronaMalik(slice, alpha, kappa, T);  % Apply filter
    filteredI(:, :, i) = slice;  % Store the filtered slice back
end
end
%%  get Region of Interest (ROI) ----

function volROIS = getROI(vols, sliceNumbers, xStart, yStart, width, height)
    % Input:
    %   vols          - input grayscale images of size M x N
    %   sliceNumbers  - slices to get the region of interest from
    %   xStart, yStart, width and height  - dimension of the patch to
    %   extract from the input images (vols).
    % Output:
    %   volROIS       - regions of interests for each slices (sliceNumbers)
    %                   and each image (vols)

N = size(vols, 2);
volROIS = cell(N, 1);

% Extract the subregion for both original and smoothed slices
for i = 1:N
    vol = vols{i};
    rois = cell(length(sliceNumbers), 1);
    for j = 1:length(sliceNumbers)
        sliceNumber = sliceNumbers(j);
        slice = vol(:, :, sliceNumber);
        
        % Get image dimensions
        [imgHeight, imgWidth] = size(slice);
        % Ensure xStart, yStart, width, and height are within image bounds
        xEnd = min(xStart(j) + width(j) - 1, imgWidth);
        yEnd = min(yStart(j) + height(j) - 1, imgHeight);
        
        % Adjust xStart and yStart if necessary
        xStartAdj = max(xStart(j), 1);
        yStartAdj = max(yStart(j), 1);
        
        % Extract the ROI considering boundary limits
        rois{j} = slice(yStartAdj:yEnd, xStartAdj:xEnd);
    end
    volROIS{i} = rois;
end
end

%% Plot Function: showROIS ----

% showROIS ----
function showROIS(noiseIndex, ...
    noisyScale,...
    titleText,....
    noisyRois,....
    gaussianFilterSigma,...
    gaussianDenoisedRois, ...
    medianSupportSize,...
    medianPadopt,...
    medianFilterDenoisedRois, ...
    sigmaD,...
    sigmaR,...
    bilateralFilterDenoisedRois, ...
    alpha,...
    kappa,...
    T,...
    peronaMalikFilterDenoisedRois,...
    fontSize)

% Create a figure and a tiled layout with two rows
figure;
t = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
title(t, titleText, 'FontSize', 24, 'FontWeight', 'bold');

% Show images for slice 102 in the first row
sliceIndex = 1;
sliceNumber = 102;
showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois, gaussianFilterSigma, ...
    gaussianDenoisedRois, medianSupportSize, medianPadopt, medianFilterDenoisedRois, ...
    sigmaD, sigmaR, bilateralFilterDenoisedRois, alpha, kappa, T,...
    peronaMalikFilterDenoisedRois, fontSize);

% Show images for slice 119 in the second row
sliceIndex = 2;
sliceNumber = 119;
showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois, gaussianFilterSigma, ...
    gaussianDenoisedRois, medianSupportSize, medianPadopt, medianFilterDenoisedRois, ...
    sigmaD, sigmaR, bilateralFilterDenoisedRois, alpha, kappa, T,...
    peronaMalikFilterDenoisedRois, fontSize);

end

% showBeforeAfter ----
function showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois, gaussianFilterSigma, ...
    gaussianDenoisedRois, medianSupportSize, medianPadopt, medianFilterDenoisedRois, sigmaD, sigmaR, ...
    bilateralFilterDenoisedRois, alpha, kappa, T, peronaMalikFilterDenoisedRois, fontSize)

% Noisy Image
nexttile;
roi = noisyRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display the original ROI from slice sliceNumber
imageText = sprintf('%d, Noisy Image\nscale:%d', sliceNumber, noisyScale);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

% Gaussian Filter
nexttile;
roi = gaussianDenoisedRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display Gaussian Filter ROI from slice sliceNumber
imageText = sprintf('%d, Gaussian Filter\nsigma:%3.2f', sliceNumber,...
    gaussianFilterSigma);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

% Median Filter
nexttile;
roi = medianFilterDenoisedRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display Median Filter ROI from sliceNumber
imageText = sprintf('%d, Median Filtering\nSupport:%s-Padding:%s', sliceNumber,...
    medianSupportSize, medianPadopt);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

% Bilateral Filter
nexttile;
roi = bilateralFilterDenoisedRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display Bilateral Filter ROI from sliceNumber
imageText = sprintf('%d, Bilateral Filter\nDomain Std:%4.2f-Range Std:%4.2f',...
    sliceNumber, sigmaD, sigmaR);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

% Perona-Malik Filter
nexttile;
roi = peronaMalikFilterDenoisedRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display Perona-Malik Filter ROI from sliceNumber
imageText = sprintf('%d, Perona-Malik Filter\nalpha:%5.2f-kappa:%5.2f,T=%d', ...
    sliceNumber, alpha, kappa, T);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

end
%% showNoisyImages ----
function showNoisyImages(noisyImages)
figure;
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
title(t, 'Noisy Slices (102,119) with Different Scales', 'FontSize', 24, 'FontWeight', 'bold');
fontSize = 14;

sliceNumber = 102;
nexttile;
vol = noisyImages{1};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 10
title('102,Scale 10', 'FontSize', fontSize,'FontWeight','bold');
nexttile;
vol = noisyImages{2};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 20
title('102, Scale 20', 'FontSize', fontSize,'FontWeight','bold');
nexttile;
vol = noisyImages{3};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 30
title('102, Scale 30', 'FontSize', fontSize,'FontWeight','bold');

sliceNumber = 119;
nexttile;
vol = noisyImages{1};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 10
title('119, Scale 10', 'FontSize', fontSize,'FontWeight','bold');
nexttile;
vol = noisyImages{2};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 20
title('119, Scale 20', 'FontSize', fontSize,'FontWeight','bold');
nexttile;
vol = noisyImages{3};
imshow(vol(:,:,sliceNumber), []);  % Display Noisy Image with scale 30
title('119, Scale 30', 'FontSize', fontSize,'FontWeight','bold');


end

%% Voxel Squared Error Difference and Plots ----
function eValues = getFilterPerformance(origVol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols)
% origVol: the original non-noisy image volume
% gaussianDenoisedVols, medianFilterDenoisedVols, ...
% bilateralFilterDenoisedVols, peronaMalikFilterDenoisedVols: different
% denoised volumes of the orignal volume using four filters:
% Gaussian, Median, Bilateral and Peron-Malik.

numNoiseLevels = size(gaussianDenoisedVols, 2);
eValues = cell(numNoiseLevels, 4);
origVol = double(origVol);
% Iterate over each noise level
for n = 1:numNoiseLevels
   % Calculate the Mean Square Error (MSE)
   smoothedI = gaussianDenoisedVols{n};
   % Compute the squared error per voxel
   E = (smoothedI - origVol).^2;
   eValues{n, 1} = E;
   smoothedI = medianFilterDenoisedVols{n};
   E = (smoothedI - origVol).^2;
   eValues{n, 2} = E;
   smoothedI = bilateralFilterDenoisedVols{n};
   E = (smoothedI - origVol).^2;
   eValues{n, 3} = E;
   smoothedI = peronaMalikFilterDenoisedVols{n};
   E = (smoothedI - origVol).^2;
   eValues{n, 4} = E;
end

end

function plotFilterPerformances(titleText, noiseLevelIndex, eValues)
figure;
fontSize = 14;
tickFontSize = 12;
t = tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
title(t, titleText, 'FontSize', 24, 'FontWeight', 'bold');

% Slice 102
sliceNumber = 102;
nexttile;
% Gaussian Filter
E = eValues{noiseLevelIndex, 1};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); % Use a colormap for visualization
colorbar;      % Add a colorbar to indicate intensity values
title(['Gaussian Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xLabelText = 'X-axis Voxel Squared Error Difference';
xlabel(xLabelText, 'FontSize', fontSize);
yLabelText = 'y-axis Voxel Squared Error Difference';
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; % Ensure the aspect ratio is correct
clim([0, max(errorSlice(:))]); 


nexttile;
% Median Filter
E = eValues{noiseLevelIndex, 2};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;      
title(['Median Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; 
clim([0, max(errorSlice(:))]); 

nexttile;
% Bilateral Filter
E = eValues{noiseLevelIndex, 3};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;     
title(['Bilateral Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image;
clim([0, max(errorSlice(:))]); 

nexttile;
% Perona-Malik Filter
E = eValues{noiseLevelIndex, 4};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;      
title(['Perona-Malik Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; 
clim([0, max(errorSlice(:))]); 

% Slice 119
sliceNumber = 119;
nexttile;
% Gaussian Filter
E = eValues{noiseLevelIndex, 1};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); % Use a colormap for visualization
colorbar;      % Add a colorbar to indicate intensity values
title(['Gaussian Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; % Ensure the aspect ratio is correct
clim([0, max(errorSlice(:))]); 

nexttile;
% Median Filter
E = eValues{noiseLevelIndex, 2};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;      
title(['Median Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; 
clim([0, max(errorSlice(:))]); 

nexttile;
% Bilateral Filter
E = eValues{noiseLevelIndex, 3};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;     
title(['Bilateral Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image;
clim([0, max(errorSlice(:))]); 

nexttile;
% Perona-Malik Filter
E = eValues{noiseLevelIndex, 4};
% Extract the slice from the error volume
errorSlice = E(:, :, sliceNumber);
imagesc(errorSlice);
colormap(jet); 
colorbar;      
title(['Perona-Malik Filter-Slice:', num2str(sliceNumber)], ...
    'FontSize', fontSize,'FontWeight','bold');
xlabel(xLabelText, 'FontSize', fontSize);
ylabel(yLabelText, 'FontSize', fontSize);
ax = gca; % Get the current axes handle
set(ax, 'FontSize', tickFontSize); % Set font size for ticks
axis image; 
clim([0, max(errorSlice(:))]); 

end


%% mse and plot ----
function mseValues = getRMSE(origVol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols)
% origVol: the original non-noisy image volume
% gaussianDenoisedVols, medianFilterDenoisedVols, ...
% bilateralFilterDenoisedVols, peronaMalikFilterDenoisedVols: different
% denoised volumes of the orignal volume using four filters:
% Gaussian, Median, Bilateral and Peron-Malik.
% 
numNoiseLevels = size(gaussianDenoisedVols, 2);
mseValues = zeros(numNoiseLevels, 4);

% Iterate over each noise level
for n = 1:numNoiseLevels
   % Calculate the Mean Square Error (MSE)
   smoothedI = gaussianDenoisedVols{n};
   mse = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   mseValues(n, 1) = mse;
   smoothedI = medianFilterDenoisedVols{n};
   mse = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   mseValues(n, 2) = mse;
   smoothedI = bilateralFilterDenoisedVols{n};
   mse = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   mseValues(n, 3) = mse;
   smoothedI = peronaMalikFilterDenoisedVols{n};
   mse = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   mseValues(n, 4) = mse;
end

end

function plotMSE(mseValues)

    % Initialize variables    
    noiseLevels = categorical({'10', '20', '30'});
    
    % Plot the MSE values
    figure('Position', [100, 100, 800, 600]);

    hold on;
    markers = {'-o', '-s', '-^', '-d'}; % Different markers for different filters
    numFilters = size(mseValues, 2);
    for f = 1:numFilters
        plot(noiseLevels, mseValues(:, f), markers{f}, 'LineWidth', 2);
    end
    hold off;
    
    % Customize the plot
    xlabel('Noise Level'); % Label for the x-axis
    ylabel('MSE');         % Label for the y-axis
    title('Mean Square Error (MSE) for Different Noise Levels and Filters',...
    'FontSize', 20,'FontWeight','bold');
    filterNames = {'Gaussian Filter', 'Median Filter', 'Bilateral Filter', 'Perona-Malik Filter'};
    legend(filterNames, 'Location', 'best'); 
    grid on;
end

%% Grid Search to Optimize Filter Parameters Using MSE

function bestParams = gridSearch(vol)
% Define parameter ranges for grid search
gaussianSigmas = [0.5, 1, 1.5, 2];
medianSupportSizes = [3, 5, 7]; % Filter sizes for the median filter
bilateralSigmaDs = [1, 2, 3]; % Spatial domain standard deviation
bilateralSigmaRs = [30, 50, 70]; % Range domain standard deviation
peronaAlphas = [0.1, 0.25, 0.5];
peronaKappas = [5, 10, 15];
peronaTs = [5, 10, 15];

% Initialize variables to store the best parameters and MSEs
bestMSE = inf(1, 4);
bestParams = cell(1, 4); % Each cell will store the best parameters for a filter

% Loop over each noise level
for noiseIdx = 1:3
    % Generate noisy image
    noisyImage = applyNoise(vol, noiseIdx * 10);
    
    % Gaussian Filter Grid Search
    for sigma = gaussianSigmas
        denoisedVol = imgaussfilt3(noisyImage, sigma);
        mse = calculateRMSE(vol, denoisedVol);
        if mse < bestMSE(1)
            bestMSE(1) = mse;
            bestParams{1} = sigma;
        end
    end
    
    % Median Filter Grid Search
    for supportSize = medianSupportSizes
        denoisedVol = medfilt3(noisyImage, [supportSize, supportSize, supportSize]);
        mse = calculateRMSE(vol, denoisedVol);
        if mse < bestMSE(2)
            bestMSE(2) = mse;
            bestParams{2} = supportSize;
        end
    end
    
    % Bilateral Filter Grid Search
    for sigmaD = bilateralSigmaDs
        for sigmaR = bilateralSigmaRs
            denoisedVol = computeBilateralFilter(noisyImage, sigmaD, sigmaR);
            mse = calculateRMSE(vol, denoisedVol);
            if mse < bestMSE(3)
                bestMSE(3) = mse;
                bestParams{3} = [sigmaD, sigmaR];
            end
        end
    end
    
    % Perona-Malik Filter Grid Search
    for alpha = peronaAlphas
        for kappa = peronaKappas
            for T = peronaTs
                denoisedVol = computePeronaMalikFilter(noisyImage, alpha, kappa, T);
                mse = calculateRMSE(vol, denoisedVol);
                if mse < bestMSE(4)
                    bestMSE(4) = mse;
                    bestParams{4} = [alpha, kappa, T];
                end
            end
        end
    end
end

% Display the best parameters for each filter
disp('Optimal Parameters and MSEs:');
disp(['Gaussian Filter: Sigma = ', num2str(bestParams{1}), ', MSE = ', num2str(bestMSE(1))]);
disp(['Median Filter: Support Size = ', num2str(bestParams{2}), ', MSE = ', num2str(bestMSE(2))]);
disp(['Bilateral Filter: SigmaD = ', num2str(bestParams{3}(1)), ', SigmaR = ', num2str(bestParams{3}(2)), ', MSE = ', num2str(bestMSE(3))]);
disp(['Perona-Malik Filter: Alpha = ', num2str(bestParams{4}(1)), ', Kappa = ', num2str(bestParams{4}(2)), ', T = ', num2str(bestParams{4}(3)), ', MSE = ', num2str(bestMSE(4))]);

end

%% Function to calculate MSE
function mse = calculateRMSE(origVol, denoisedVol)
    mse = sqrt(mean((origVol(:) - denoisedVol(:)).^2));
end


