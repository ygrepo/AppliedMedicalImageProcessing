clearvars 
vol = niftiread('data/sub-11_T1w.nii');
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
scale10 = 10;
volscale10 = applyNoise(vol, scale10);
scale20 = 20;
volscale20 = applyNoise(vol, scale20);
scale30 = 30;
volscale30 = applyNoise(vol, scale30);
%% Gaussian filtering of noisy images ----
sigmaGaussianFilter = .1;
gaussianDenoisedVols = {};
gaussianDenoisedVols{1} = imgaussfilt3(volscale10, sigmaGaussianFilter);
gaussianDenoisedVols{2} = imgaussfilt3(volscale20, sigmaGaussianFilter);
gaussianDenoisedVols{3} = imgaussfilt3(volscale30, sigmaGaussianFilter);
%% Median Filter ----
medianFilterDenoisedVols = {};
medianFilterDenoisedVols{1} = medfilt3(volscale10);
medianFilterDenoisedVols{2} = medfilt3(volscale20);
medianFilterDenoisedVols{3} = medfilt3(volscale30);
%% BilateralFilterGray ----
sigmaD = 2;  % Standard deviation for the domain (spatial) Gaussian kernel
sigmaR = 25; % Standard deviation for the range Gaussian kernel
bilateralFilterDenoisedVols = {};
bilateralFilterDenoisedVols{1} =computeBilateralFilter(volscale10, sigmaD, sigmaR);
bilateralFilterDenoisedVols{2} =computeBilateralFilter(volscale20, sigmaD, sigmaR);
bilateralFilterDenoisedVols{3} =computeBilateralFilter(volscale30, sigmaD, sigmaR);
%% Perona-Malik Filter ----
alpha = 0.25;   % Update rate
kappa = 15;     % Smoothness parameter
T = 10;         % Number of iterations
peronaMalikFilterDenoisedVols = {};
peronaMalikFilterDenoisedVols{1} =computePeronaMalikFilter(volscale10, alpha, kappa, T);
peronaMalikFilterDenoisedVols{2} =computePeronaMalikFilter(volscale20, alpha, kappa, T);
peronaMalikFilterDenoisedVols{3} =computePeronaMalikFilter(volscale30, alpha, kappa, T);


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


%% computeBilateralFilter ----
function filteredI =computeBilateralFilter(I, sigmaD, sigmaR)

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