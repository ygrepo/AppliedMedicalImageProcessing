clearvars 
vol = niftiread('data/sub-11_T1w.nii');
%%
sliceViewer(vol);
% montage(vol)
% title('Original image volume')
%volumeViewer(V)
%% Generate noisy images ----
scale10 = 10;
volscale10 = applyNoise(vol, scale10);
scale20 = 20;
volscale20 = applyNoise(vol, scale20);
scale30 = 30;
volscale30 = applyNoise(vol, scale30);
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
imshow(volscale10(:,:,sliceNumber), []);
title('Slice 102');
hold off;
%% Gaussian smoothing Filter, sigma 10 ----
sigmaGaussianFilter = .1;
volSmooth = imgaussfilt3(volscale10, sigmaGaussianFilter);
sliceNumber = 102;
volsigmaSlice102 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale10)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale10)],...
    roi_119,...
    ['Slice 102, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)], ...
    roi_102_smoothed,...
    ['Slice 119, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)],...
    roi_119_smoothed);
%% Gaussian smoothing Filter, sigma 20 ----
sigmaGaussianFilter = .1;
volSmooth = imgaussfilt3(volscale20, sigmaGaussianFilter);
sliceNumber = 102;
volsigmaSlice102 = volscale20(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale20(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale20)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale20)],...
    roi_119,...
    ['Slice 102, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)], ...
    roi_102_smoothed,...
    ['Slice 119, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)],...
    roi_119_smoothed);

%% Gaussian smoothing Filter, sigma 30 ----
sigmaGaussianFilter = .1;
volSmooth = imgaussfilt3(volscale30, sigmaGaussianFilter);
sliceNumber = 102;
volsigmaSlice102 = volscale30(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale30(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale30)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale30)],...
    roi_119,...
    ['Slice 102, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)], ...
    roi_102_smoothed,...
    ['Slice 119, smoothed with Gaussian filter, sigma:', num2str(sigmaGaussianFilter)],...
    roi_119_smoothed);

%% Median Filter ----
volSmooth = medfilt3(volscale10);
sliceNumber = 102;
volsigmaSlice102 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale30)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale30)],...
    roi_119,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_102_smoothed,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_119_smoothed);
%% Bilateral Filter, sigma 10 ----
sigmaD = 2;  % Standard deviation for the domain (spatial) Gaussian kernel
sigmaR = 25; % Standard deviation for the range Gaussian kernel

[~, ~, dim3] = size(volscale10); 
volSmooth = zeros(size(volscale10));  % Initialize an array to store the filtered image

for i = 1:dim3
    slice = volscale10(:, :, i);  % Extract the ith slice
    slice = BilateralFilterGraySep(slice, sigmaD, sigmaR);  % Apply median filter
    volSmooth(:, :, i) = slice;  % Store the filtered slice back
end
%% Bilateral Filter, sigma 10 ----
sliceNumber = 102;
volsigmaSlice102 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale30)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale30)],...
    roi_119,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_102_smoothed,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_119_smoothed);

%% Perona-Malik Filter, sigma 10 ----
[~, ~, dim3] = size(volscale10); 
volSmooth = zeros(size(volscale10));  % Initialize an array to store the filtered image
alpha = 0.25;   % Update rate
kappa = 15;     % Smoothness parameter
T = 10;         % Number of iterations
for i = 1:dim3
    slice = volscale10(:, :, i);  % Extract the ith slice
    slice = PeronaMalik(slice, alpha, kappa, T);  % Apply median filter
    volSmooth(:, :, i) = slice;  % Store the filtered slice back
end
%% Perona-Malik Filter, sigma 10 ----
sliceNumber = 102;
volsigmaSlice102 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice102 = volSmooth(:,:,sliceNumber);
sliceNumber = 119;
volsigmaSlice119 = volscale10(:,:,sliceNumber);
volsigmaSmoothedSlice119 = volSmooth(:,:,sliceNumber);
% Define the region of interest (ROI)
x_start = 50;
y_start = 100;
width = 75;
height = 75;

% Extract the subregion for both original and smoothed slices
roi_102 = volsigmaSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_102_smoothed = volsigmaSmoothedSlice102(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119 = volsigmaSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);
roi_119_smoothed = volsigmaSmoothedSlice119(y_start:y_start+height-1,...
    x_start:x_start+width-1);

showSlices(['Slice 102, Original Noisy Image, scale:', num2str(scale30)],...
    roi_102,...
    ['Slice 119, Original Noisy Image, scale:', num2str(scale30)],...
    roi_119,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_102_smoothed,...
    'Slice 102, smoothed with Median filter (3x3x3)', ...
    roi_119_smoothed);

%% applyNoise ----
function Innoisy = applyNoise(I, s)
I = flipdim (permute(I, [2 1 3]), 1);
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

%%
% [~, ~, dim3] = size(vol);  % Get dimensions of the 3D image
% sigmaGaussianFilter = .1;  % Define the standard deviation for Gaussian filter
% volSmooth = zeros(size(vol));  % Initialize an array to store the filtered image
% 
% for i = 1:dim3
%     slice = vol(:, :, i);  % Extract the ith slice
%     filteredSlice = imgaussfilt(slice, sigmaGaussianFilter);  % Apply Gaussian filter
%     volSmooth(:, :, i) = filteredSlice;  % Store the filtered slice back
% end
% 
% figure
% sliceNumber = 102;
% montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
% title("Original Slice (Left) Vs. Gaussian Filtered Slice (Right)")
% 
% 
% %% Median Filter ----
% volSmooth = medfilt3(vol);
% sliceNumber = 102;
% montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
% title("Original Slice (Left) Vs. Median Filter Slice (Right)")
% 
% %%
% windowSize = 3;  % Define the window size for the median filter
% [dim1, dim2, dim3] = size(vol); 
% volSmooth = zeros(size(vol));  % Initialize an array to store the filtered image
% 
% for i = 1:dim3
%     slice = vol(:, :, i);  % Extract the ith slice
%     %filteredSlice = medfilt2(slice);  % Apply median filter
%     filteredSlice = medfilt2(slice, [windowSize windowSize]);  % Apply median filter
%     volSmooth(:, :, i) = filteredSlice;  % Store the filtered slice back
% end
% sliceNumber = 15;
% montage({vol(:,:,sliceNumber),volSmooth(:,:,sliceNumber)})
% title("Original Slice (Left) Vs. Median Filter Slice (Right)")