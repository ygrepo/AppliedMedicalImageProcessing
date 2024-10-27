%%
% vol = load('mri');
% %siz = vol.siz;
% vol = squeeze(vol.D);   
% sigma = 2;
% 
% volSmooth = imgaussfilt3(vol, sigma);
% figure
% montage(vol)
% figure
% 
% montage(reshape(volSmooth,siz(1),siz(2),1,siz(3)))
% % title('Gaussian filtered image volume')

%%
% clearvars
% clc
% load mristack
% alpha = 0.25;   % Update rate
% kappa = 15;     % Smoothness parameter
% T = 10;         % Number of iterations
% diffusedImage = computePeronaMalikFilter(mristack, alpha, kappa, T);
% imshowpair(mristack(:,:,10),diffusedImage(:,:,10),'montage')
% title('Noisy Image (Left) vs. Anisotropic-Diffusion-Filtered Image (Right)')
%%
% clc
% I = imread('cameraman.tif');
% %imshow(I)
% sigmaD = 2;
% sigmaR = 50;
% J = computeBilateralFilter(I, sigmaD, sigmaR);
% imshowpair(I,J,'montage')
%%
clearvars
clc
vol = niftiread('data/sub-11_T1w.nii.gz');
vol = flip (permute(vol, [2 1 3]), 1);

%%
figure
imshow(double(vol(:,:,102)), []);
%% applyNoise ----
% I = double (vol(:,:,102));
% dim = size(I);
% s = 10;
% x = s .* randn(dim) + I;
% y = s .* randn(dim);
% Innoisy = sqrt(x.^2+y.^2);
% figure
% imshow(Innoisy, []);

%% Look at Slice 102 and Slice 119 ---
%sliceViewer(volsigma10);
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

%% Generate noisy images ----
noisyImages = {};
noisyImages{1} = applyNoise(vol, 10);
noisyImages{2} = applyNoise(vol, 20);
noisyImages{3} = applyNoise(vol, 30);
showNoisyImages(noisyImages);
%%
% alpha = 0.25;   % Update rate
% kappa = 15;     % Smoothness parameter
% T = 10;         % Number of iterations
% I = noisyImages{1};
% diffusedImage = computePeronaMalikFilter(I, alpha, kappa, T);
% imshowpair(I(:,:,119),diffusedImage(:,:,119),'montage')
% title('Noisy Image (Left) vs. Anisotropic-Diffusion-Filtered Image (Right)')
%%
% sigmaD = 2;  % Standard deviation for the domain (spatial) Gaussian kernel
% sigmaR = 50; % Standard deviation for the range Gaussian kernel
% I = noisyImages{1};
% diffusedImage = computeBilateralFilter(I, sigmaD, sigmaR);
% imshowpair(I(:,:,119),diffusedImage(:,:,119),'montage')
% title('Noisy Image (Left) vs. Bilateral-Filtered Image (Right)')
%%
% Gaussian filtering of noisy images ----
gaussianFilterSigma = 1;
gaussianDenoisedVols = {};
gaussianDenoisedVols{1} = imgaussfilt3(noisyImages{1}, gaussianFilterSigma);
gaussianDenoisedVols{2} = imgaussfilt3(noisyImages{2}, gaussianFilterSigma);
gaussianDenoisedVols{3} = imgaussfilt3(noisyImages{3}, gaussianFilterSigma);
%%
% Median Filter of noisy images ----
medianFilterDenoisedVols = {};
medianFilterDenoisedVols{1} = medfilt3(noisyImages{1});
medianFilterDenoisedVols{2} = medfilt3(noisyImages{2});
medianFilterDenoisedVols{3} = medfilt3(noisyImages{3});
%%
% BilateralFilterGray of noisy images ----
sigmaD = 2;  % Standard deviation for the domain (spatial) Gaussian kernel
sigmaR = 51; % Standard deviation for the range Gaussian kernel
bilateralFilterDenoisedVols = {};
bilateralFilterDenoisedVols{1} =computeBilateralFilter(noisyImages{1}, sigmaD, sigmaR);
bilateralFilterDenoisedVols{2} =computeBilateralFilter(noisyImages{2}, sigmaD, sigmaR);
bilateralFilterDenoisedVols{3} =computeBilateralFilter(noisyImages{3}, sigmaD, sigmaR);
%%
% Perona-Malik Filter of noisy images ----
alpha = 0.25;   % Update rate
kappa = 15;     % Smoothness parameter
T = 10;         % Number of iterations
peronaMalikFilterDenoisedVols = {};
peronaMalikFilterDenoisedVols{1} =computePeronaMalikFilter(noisyImages{1}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{2} =computePeronaMalikFilter(noisyImages{2}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{3} =computePeronaMalikFilter(noisyImages{3}, alpha, kappa, T);
%% Define the region of interest (ROI) ----
xStart = [80, 110];
yStart = [50, 80];
% xStart = [80, 110];
% yStart = [50, 80];
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
%%
clc
figure
I = noisyRois{1};
%sliceViewer(noisyImages{1});
imshowpair(I{1},I{2},'montage')
title('Noisy ROI, 102, 119')

%% Before and After Filtering, Noise scale = 10 ----
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
%% Before and After Filtering, Noise scale = 20 ----
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
%% Before and After Filtering, Noise scale = 30 ----
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
%%

slices = [102,119];
quantifyFilterPerformance(gaussianDenoisedVols{1}, vol, slices)

%%
function quantifyFilterPerformance(smoothedI, origI, slices)
    % Check that the input volumes have the same size
    if ~isequal(size(smoothedI), size(origI))
        error('The dimensions of smoothedI and origI must match.');
    end
    
    % Compute the squared error per voxel
    E = (smoothedI - double(origI)).^2;
    
    % Display the squared error for each of the specified slices
    for i = 1:length(slices)
        sliceNumber = slices(i);
        
        % Extract the slice from the error volume
        errorSlice = E(:, :, sliceNumber);
        
        % Display the error slice using imagesc
        figure;
        imagesc(errorSlice);
        colormap(jet); % Use a colormap for visualization
        colorbar;      % Add a colorbar to indicate intensity values
        title(['Squared Error - Slice ', num2str(sliceNumber)]);
        xlabel('X-axis');
        ylabel('Y-axis');
        axis image; % Ensure the aspect ratio is correct
       
        clim([0, max(errorSlice(:))]); % Optional: set a range based on max error
    end
end
%%
plotMSE(double(vol), noisyImages, gaussianDenoisedVols)
%%
function plotMSE(origI, noiseLevels, filters)

end

function plotMSE2(origI, noiseLevels, filters)
    % origI: the original non-noisy image volume
    % noiseLevels: a cell array containing image volumes with different noise levels
    % filters: a cell array of function handles for the filters to apply

    % Initialize variables
    numNoiseLevels = length(noiseLevels);
    numFilters = length(filters);
    mseValues = zeros(numNoiseLevels, numFilters);
    
    % Iterate over each noise level
    for n = 1:numNoiseLevels
        noisyVol = noiseLevels{n};
        
        % Iterate over each filter
        for f = 1:numFilters
            filterFunc = filters{f};
            
            % Apply the filter to the noisy volume
            I_smooth = filterFunc(noisyVol);
            
            % Calculate the Mean Square Error (MSE)
            mse = sqrt(mean((I_smooth(:) - origI(:)).^2));
            mseValues(n, f) = mse;
        end
    end
    
    % Plot the MSE values
    figure;
    hold on;
    markers = {'-o', '-s', '-^', '-d'}; % Different markers for different filters
    for f = 1:numFilters
        plot(1:numNoiseLevels, mseValues(:, f), markers{f}, 'LineWidth', 2);
    end
    hold off;
    
    % Customize the plot
    xlabel('Noise Level');
    ylabel('MSE');
    title('Mean Square Error (MSE) for Different Noise Levels and Filters');
    legend({'Filter 1', 'Filter 2', 'Filter 3', 'Filter 4'}, 'Location', 'best');
    grid on;
end


%% applyNoise ----
function Innoisy = applyNoise(I, s)
% I = flip(permute(I, [2 1 3]), 1);  % Adjust orientation if needed
% I = double(I);
% dim = size(I);
% Innoisy = I + s * randn(dim);  % Add Gaussian noise directly
% I = flip (permute(I, [2 1 3]), 1);
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
function I_filtered = BilateralFilterGrayOptimized(I, sigma_d, sigma_r)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigma_d  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigma_r  - standard deviation for the range Gaussian kernel
    % Output:
    %   I_filtered - output filtered grayscale image of size M x N
    
    % Convert image to double for precision
    I = double(I);
    
    % Get the size of the image
    [M, N] = size(I);
    
    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigma_d);
    
    % Create a grid of distances (m^2 + n^2) for the spatial Gaussian (domain kernel)
    [X, Y] = meshgrid(-K:K, -K:K);
    spatial_weights = exp(-(X.^2 + Y.^2) / (2 * sigma_d^2));
    
    % Initialize the output image
    I_filtered = zeros(M, N);
    
    % Pad the image to handle borders
    padded_I = padarray(I, [K K], 'symmetric');
    
    % For each pixel in the image
    for u = 1:M
        for v = 1:N
            % Extract the local neighborhood around pixel (u,v)
            local_region = padded_I(u:u+2*K, v:v+2*K);
            
            % Compute the range Gaussian weights based on intensity differences
            intensity_diff = local_region - I(u,v);
            range_weights = exp(-(intensity_diff.^2) / (2 * sigma_r^2));
            
            % Compute the combined bilateral weights
            bilateral_weights = spatial_weights .* range_weights;
            
            % Normalize the weights
            normalized_weights = bilateral_weights / sum(bilateral_weights(:));
            
            % Compute the filtered intensity as the weighted sum of the local neighborhood
            I_filtered(u, v) = sum(sum(normalized_weights .* local_region));
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
%filteredI = imbilatfilt(I, sigmaD, sigmaR);
%filteredI = BilateralFilterGraySep(I, sigmaD, sigmaR);
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

%% Plot Functions ----

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

%% showSlices ----
function showSlices(titletext_ROI102,...
    roi_102,...
    titletext_ROI119,...
    roi_119,...
    titletext_smoothedROI102, ...
    roi_102_smoothed,...
    titletext_smoothedROI119,...
    roi_119_smoothed)

figure;
t = tiledlayout(2, 2);
title(t, 'Original Image Before and After Filtering','FontSize', 24,'FontWeight','bold');
% hold on 
fontSize = 14;


% Display original and smoothed slice 102
nexttile;
imshow(roi_102, []);  % Display the original ROI from slice 102
title(titletext_ROI102, 'FontSize', fontSize,'FontWeight','bold');

nexttile;
imshow(roi_102_smoothed, []);  % Display the smoothed ROI from slice 102
title(titletext_smoothedROI102,...
    'FontSize', fontSize,'FontWeight','bold');

% % Display original and smoothed slice 119
nexttile;
imshow(roi_119, []);  % Display the original ROI from slice 119
title(titletext_ROI119, 'FontSize', fontSize,'FontWeight','bold');


nexttile;
imshow(roi_119_smoothed, []);  % Display the smoothed ROI from slice 119
title(titletext_smoothedROI119,...
    'FontSize', fontSize,'FontWeight','bold');


end

