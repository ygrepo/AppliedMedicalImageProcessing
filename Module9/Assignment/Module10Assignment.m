
clearvars
clc
vol = niftiread('data/sub-11_T1w-1.nii.gz');
vol = flip (permute(vol, [2 1 3]), 1);

% bestParams = gridSearch(vol);
% sigmaG = 1;
% sigmaS = 2;
% T = 10;
% dt = 0.2;
% a1 = 0.5;
% a2 = 0.9;

% Generate noisy images ----
noisyImages = {};
noisyImages{1} = applyNoise(vol, 10);
noisyImages{2} = applyNoise(vol, 20);
noisyImages{3} = applyNoise(vol, 30);
showNoisyImages(noisyImages)

% Geometry Preserving Anisotropic Diffusion Filter of noisy images ----
sigmaG = 2;
sigmaS = 1;
T = 10;
dt = 0.2;
a1 = 0.5;
a2 = 0.9;
geometryPreservingAnisotropicDiffFilterDenoisedVols = {};
geometryPreservingAnisotropicDiffFilterDenoisedVols{1} = ...
    computeGeometryPreservingAnisotropicDiffFilter(noisyImages{1},...
    T, dt, sigmaG, sigmaS, a1, a2);  % Apply  filter
geometryPreservingAnisotropicDiffFilterDenoisedVols{2} = ...
    computeGeometryPreservingAnisotropicDiffFilter(noisyImages{2},...
    T, dt, sigmaG, sigmaS, a1, a2);  % Apply  filter
geometryPreservingAnisotropicDiffFilterDenoisedVols{3} = ...
    computeGeometryPreservingAnisotropicDiffFilter(noisyImages{3},...
    T, dt, sigmaG, sigmaS, a1, a2);  % Apply  filter

% Define the region of interest (ROI) ----
xStart = [80, 110];
yStart = [50, 80];
width = [50, 50];
height = [50, 50];
sliceNumbers = [102, 119];
noisyRois = getROI(noisyImages, sliceNumbers, xStart, yStart, width, height);
geometryPreservingAnisotropicDiffFilterdRois = getROI(geometryPreservingAnisotropicDiffFilterDenoisedVols,... 
    sliceNumbers, xStart, yStart, width, height);

% Before and After Filtering, Noise scale = 10 ----
noiseIndex = 1;
scaleValue = 10;
fontSize = 14;
titleText = sprintf('Noise with Scale %d, Before and After Filtering ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    sigmaS,....
    sigmaG,...
    T,...
    dt,...
    geometryPreservingAnisotropicDiffFilterdRois,...
    fontSize);

% Before and After Filtering, Noise scale = 20 ----
noiseIndex = 2;
scaleValue = 20;
fontSize = 14;
titleText = sprintf('Noise with Scale %d, Before and After Filtering ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    sigmaS,....
    sigmaG,...
    T,...
    dt,...
    geometryPreservingAnisotropicDiffFilterdRois,...
    fontSize);
% Before and After Filtering, Noise scale = 30 ----
noiseIndex = 3;
scaleValue = 30;
fontSize = 14;
titleText = sprintf('Noise with Scale %d, Before and After Filtering ROIS in Slices 102 and 119', scaleValue);
showROIS(noiseIndex, ...
    scaleValue,...
    titleText,....
    noisyRois,....
    sigmaS,....
    sigmaG,...
    T,...
    dt,...
    geometryPreservingAnisotropicDiffFilterdRois,...
    fontSize);

% Voxel Squared Error and Plot ----
eValues = getVoxelSquaredErrorDifference(vol,...
    geometryPreservingAnisotropicDiffFilterDenoisedVols);
% Noise Level 10, 20 and 30 and Voxel Squared Error - Slices (102,119) ---
plotVoxelSquaredErrorDifference(eValues);

% Comparison with Gaussian, Median, Bilateral and Perona-Malik Filters.
clc
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
sigmaS = 1;  % Standard deviation for the domain (spatial) Gaussian kernel
sigmaR = 30; % Standard deviation for the range Gaussian kernel
bilateralFilterDenoisedVols = {};
bilateralFilterDenoisedVols{1} =computeBilateralFilter(noisyImages{1}, sigmaS, sigmaR);
bilateralFilterDenoisedVols{2} =computeBilateralFilter(noisyImages{2}, sigmaS, sigmaR);
bilateralFilterDenoisedVols{3} =computeBilateralFilter(noisyImages{3}, sigmaS, sigmaR);

% Perona-Malik Filter of noisy images ----
alpha = 0.1;   % Update rate
kappa = 15;     % Smoothness parameter
T = 10;         % Number of iterations
peronaMalikFilterDenoisedVols = {};
peronaMalikFilterDenoisedVols{1} =computePeronaMalikFilter(noisyImages{1}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{2} =computePeronaMalikFilter(noisyImages{2}, alpha, kappa, T);
peronaMalikFilterDenoisedVols{3} =computePeronaMalikFilter(noisyImages{3}, alpha, kappa, T);
% MSE and Plot ----
mseValues = getRMSE(vol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols,...
    geometryPreservingAnisotropicDiffFilterDenoisedVols);
plotMSE(mseValues);

%% Geometry Preserving Anisotropic Diffusion Filter ----
function filteredI = computeGeometryPreservingAnisotropicDiffFilter(I,...
    T, dt, sigmaS, sigmaG, a1, a2)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    fprintf('Slice:%d\n', i)
    slice = double(I(:, :, i));
    slice(:,:,1) =slice;
    slice = GeometryPreservingAnisotropicDiffFilter(slice,...
        T, dt, sigmaS, sigmaG, a1, a2);  % Apply  filter
    slice = squeeze(slice);
    filteredI(:, :, i) = slice;  % Store the filtered slice back
 end
end

function I = GeometryPreservingAnisotropicDiffFilter(I,...
    T, dt, sigmaG, sigmaS, a1, a2)
    % Input:
    % I: input image of size M x N with K channels
    % T: number of iterations
    % dt: time increment
    % sigmaG: width of Gaussian for smoothing the gradient
    % sigmaS: width of Gaussian for smoothing the structure matrix
    % a1, a2: diffusion parameters for directions of min/max variation
    % Output:
    %   I - output filtered grayscale image of size M x N

% Define the size of the image
[M, N, K] = size(I);

% Initialize auxiliary matrices
D = zeros(K, M, N, 2);        % Gradient maps for each channel
H = zeros(K, M, N, 2, 2);       % Hessian

% Iterate for T steps
for t = 1:T    
    % Compute gradient maps D and structure matrix H for each channel
    for k = 1:K
        %fprintf('T:%d-Channel:%d\n', t, k)
        % Compute the gradient maps for the entire channel k at once
        [Ix_smoothed, Iy_smoothed] = computeGradient(I(:, :, k), sigmaG);
        
        % Store the gradients in D for the current channel
        D(k, :, :, 1) = Ix_smoothed;  % Gradient in x-direction
        D(k, :, :, 2) = Iy_smoothed;  % Gradient in y-direction
        
        % Compute the Hessian matrix H for the entire channel k at once
        H_k = computeHessian(I(:, :, k));  % Call with the k-th channel slice
        
        % Store the Hessian matrix for channel k in the overall H matrix
        H(k, :, :, :, :) = H_k;
    end
     
    % Calculate the structure matrix G by summing over all channels K
    G = zeros(M, N, 2, 2);  % Initialize G
    for k = 1:K
        G(:, :, 1, 1) = G(:, :, 1, 1) + squeeze(D(k, :, :, 1)).^2;  % G0 = sum_k (Dx^2)
        G(:, :, 1, 2) = G(:, :, 1, 2) + squeeze(D(k, :, :, 1)) .* squeeze(D(k, :, :, 2));  % G1 = sum_k (Dx * Dy)
        G(:, :, 2, 1) = G(:, :, 1, 2);  % Symmetric G1
        G(:, :, 2, 2) = G(:, :, 2, 2) + squeeze(D(k, :, :, 2)).^2;  % G2 = sum_k (Dy^2)
    end

    % % Smooth each component of the structure matrix G
    G(:, :, 1, 1) = imgaussfilt(G(:, :, 1, 1), sigmaS);  % Smooth G0
    G(:, :, 1, 2) = imgaussfilt(G(:, :, 1, 2), sigmaS);  % Smooth G1
    G(:, :, 2, 1) = G(:, :, 1, 2);                       % Ensure symmetry
    G(:, :, 2, 2) = imgaussfilt(G(:, :, 2, 2), sigmaS);  % Smooth G2
    A=computeA(G, a1, a2); % Geometry matrix
    
    % Calculate B and beta_max for the current iteration
    [B, betaMax] = computeBAndBetaMax(A, H, K, M, N);

    % Calculate alpha based on dt and beta_max
    alpha = dt / betaMax;
    % Update I for all channels at once
    for k = 1:K
        I(:,:,k)  = I(:,:,k) + permute(alpha * B(k,:,:), [2 3 1]);
    end
end
end

function [Ix_smoothed, Iy_smoothed] = computeGradient(I, sigmaG)
    % Compute the gradients across the whole image
    
    a = (2 - sqrt(2)) / 4; % Good rotation invariance
    b = (sqrt(2) - 1) / 2; % Good rotation invariance

    % Define convolution kernels H_x and H_y
    Hx = [-a, 0, a;
          -b, 0, b;
          -a, 0, a];
    
    Hy = [-a, -b, -a;
           0,  0,  0;
           a,  b,  a];
    
    % Convolve the whole image with Hx and Hy to get the gradient maps
    Ix = conv2(I, Hx, 'same');  % Gradient in x-direction
    Iy = conv2(I, Hy, 'same');  % Gradient in y-direction
    % Apply Gaussian smoothing to the gradient components if required
    Ix_smoothed = imgaussfilt(Ix, sigmaG);
    Iy_smoothed = imgaussfilt(Iy, sigmaG);
end

function H = computeHessian(I_channel)
    % computeHessian Computes the Hessian matrix for a 2D image or image channel
    % I_channel: 2D input image for a single channel (MxN)
    % H: output Hessian matrix of size (M x N x 2 x 2)
    %    H(u, v, 1, 1) = D2I/Dx2 at pixel (u, v)
    %    H(u, v, 1, 2) = D2I/DxDy at pixel (u, v)
    %    H(u, v, 2, 1) = D2I/DxDy at pixel (u, v)
    %    H(u, v, 2, 2) = D2I/Dy2 at pixel (u, v)
    
    % Define the Hessian kernels based on the given specifications
    Hxx_kernel = [1, -2, 1];          % 1x3 kernel for second derivative in x
    Hyy_kernel = [1; -2; 1];          % 3x1 kernel for second derivative in y
    Hxy_kernel = [1/4, 0, -1/4;       % 3x3 kernel for mixed second derivative
                  0,   0,  0;
                 -1/4, 0,  1/4];
    
    % Compute the second-order partial derivatives using convolution
    Hxx = conv2(I_channel, Hxx_kernel, 'same');  % D2I/Dx2
    Hyy = conv2(I_channel, Hyy_kernel, 'same');  % D2I/Dy2
    Hxy = conv2(I_channel, Hxy_kernel, 'same');  % D2I/DxDy (mixed derivative)
    
    % Initialize the output Hessian matrix H
    [M, N] = size(I_channel);
    H = zeros(M, N, 2, 2);
    
    % Populate the Hessian matrix
    H(:, :, 1, 1) = Hxx;  % Hxx (D2I/Dx2)
    H(:, :, 1, 2) = Hxy;  % Hxy (D2I/DxDy)
    H(:, :, 2, 1) = Hxy;  % Hxy (D2I/DxDy)
    H(:, :, 2, 2) = Hyy;  % Hyy (D2I/Dy2)
end

function A=computeA(G, a1, a2)
% Initialize matrix A for the entire image
[M, N, ~, ~] = size(G);
A = zeros(M, N, 2, 2);

% Loop over each pixel in the image
for u = 1:M
    for v = 1:N
        % Extract the structure matrix at pixel (u, v)
        G_uv = squeeze(G(u, v, :, :));  % G is a 2x2 matrix at (u, v)
        
        % Calculate eigenvalues and eigenvectors of G_uv using eig
        [V, D] = eig(G_uv);
        
        % Extract the eigenvalues
        lambda1 = max(D(1,1), D(2,2)); % Lambda1 is the larger eigenvalue
        lambda2 = min(D(1,1), D(2,2)); % Lambda2 is the smaller eigenvalue
        
        % Ensure the eigenvector corresponding to lambda1 is used as e1
        if D(1,1) >= D(2,2)
            e1 = V(:, 1);  % Eigenvector corresponding to lambda1
        else
            e1 = V(:, 2);  % Eigenvector corresponding to lambda1
        end

        % Extract components of e1
        x1 = e1(1);
        y1 = e1(2);

        % Normalize eigenvectors e1
        normE1 = sqrt(x1.^2 + y1.^2);
        x1 = x1 ./ normE1;
        y1 = y1 ./ normE1;
    
        % Calculate coefficients c1 and c2
        c1 = 1 / (1 + lambda1 + lambda2)^a1;
        c2 = 1 / (1 + lambda1 + lambda2)^a2;
        
        % Construct the diffusion matrix A at (u, v)
        A(u, v, 1, 1) = c1 * y1^2 + c2 * x1^2;
        A(u, v, 1, 2) = (c2 - c1) * x1 * y1;
        A(u, v, 2, 1) = A(u, v, 1, 2);  % Symmetric
        A(u, v, 2, 2) = c1 * x1^2 + c2 * y1^2;
    end
end
end

function [B, beta_max] = computeBAndBetaMax(A, H, K, M, N)
    % Initialize B
    B = zeros(K, M, N);

    % Vectorized trace computation for each channel
    for k = 1:K
        % Compute B(k, :, :) = trace(A * H) across the whole image
        A11 = A(:, :, 1, 1);
        A12 = A(:, :, 1, 2);
        A22 = A(:, :, 2, 2);

        H11 = squeeze(H(k, :, :, 1, 1));
        H12 = squeeze(H(k, :, :, 1, 2));
        H22 = squeeze(H(k, :, :, 2, 2));

        % Calculate the trace
        B(k, :, :) = A11 .* H11 + 2 * A12 .* H12 + A22 .* H22;
    end

    % Compute beta_max as the maximum absolute value in B
    beta_max = max(abs(B(:)));
end
%% BilateralFilterGray ----
function filteredI = BilateralFilterGraySep(I, sigmaS, sigmaR)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigmaS  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigmaR  - standard deviation for the range Gaussian kernel
    % Output:
    %   filteredI - output filtered grayscale image of size M x N

    % Convert image to double for precision
    I = double(I);

    % Get the size of the image
    [M, N] = size(I);

    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigmaS);

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
                    wd = exp(-m^2 / (2 * sigmaS^2));

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
                    wd = exp(-n^2 / (2 * sigmaS^2));

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
function filteredI = computeBilateralFilter(I, sigmaS, sigmaR)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    slice = I(:, :, i);  % Extract the ith slice
    slice = BilateralFilterGraySep(slice, sigmaS, sigmaR);  % Apply  filter
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

%% applyNoise ----
function Innoisy = applyNoise(I, s)
% Generate a noisy image using Rician distribution
I = double (I);
dim = size(I);
x = s .* randn(dim) + I;
y = s .* randn(dim);
Innoisy = sqrt(x.^2+y.^2);
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
    sigmaS,....
    sigmaG,...
    T,...
    dt,...
    geometryPreservingAnisotropicDiffFilterdRois,...
    fontSize)

% Create a figure and a tiled layout with two rows
figure;
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
title(t, titleText, 'FontSize', 24, 'FontWeight', 'bold');

% Show images for slice 102 in the first row
sliceIndex = 1;
sliceNumber = 102;
showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois, ...
    sigmaS, sigmaG, T, dt,...
    geometryPreservingAnisotropicDiffFilterdRois, fontSize);

% Show images for slice 119 in the second row
sliceIndex = 2;
sliceNumber = 119;
showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois, ...
    sigmaS, sigmaG, T, dt,...
    geometryPreservingAnisotropicDiffFilterdRois, fontSize);
end

% showBeforeAfter ----
function showBeforeAfter(noiseIndex, sliceIndex, sliceNumber, noisyScale, noisyRois,...
        sigmaS, sigmaG, T, dt, geometryPreservingAnisotropicDiffFilterdRois,...
        fontSize)

% Noisy Image
nexttile;
roi = noisyRois{noiseIndex};
roi = roi{sliceIndex};
imshow(roi, []);  % Display the original ROI from slice sliceNumber
imageText = sprintf('%d, Noisy Image\nscale:%d', sliceNumber, noisyScale);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

% Geometry Preserving Anisotropic Diffusion Filter 
nexttile;
roi = geometryPreservingAnisotropicDiffFilterdRois{noiseIndex};
roi = roi{sliceIndex};
% Display Geometry Preserving Anisotropic Diffusion Filter ROI from slice sliceNumber
imshow(roi, []);  
ImgText = '%d, Geometry Preserving Anisotropic Diffusion Filter\nsigmaS:%3.2f-sigmaG:%3.2f-T:%d-dt:%3.2f';
imageText = sprintf(ImgText, sliceNumber,...
    sigmaS, sigmaG, T, dt);
title(imageText, 'FontSize', fontSize,'FontWeight','bold');

end
%% Voxel Squared Error Difference and Plots ----
function eValues = getVoxelSquaredErrorDifference(origVol,....
    geometryPreservingAnisotropicDiffFilterDenoisedVols)
% origVol: the original non-noisy image volume
% geometryPreservingAnisotropicDiffFilterDenoisedVols: different
% denoised volumes of the orignal volume using 
% Geometry Preserving Anisotropic Diffusion Filter.

numNoiseLevels = size(geometryPreservingAnisotropicDiffFilterDenoisedVols, 2);
eValues = cell(numNoiseLevels, 1);
origVol = double(origVol);
% Iterate over each noise level
for n = 1:numNoiseLevels
   % Compute the squared error per voxel
   smoothedI = geometryPreservingAnisotropicDiffFilterDenoisedVols{n};
   eValues{n} = (smoothedI - origVol).^2;
end

end

function plotVoxelSquaredErrorDifference(eValues)
    fontSize = 14;
    tickFontSize = 12;
    xLabelText = 'X-axis Voxel Squared Error Difference';
    yLabelText = 'Y-axis Voxel Squared Error Difference';
    sliceNumbers = [102, 119]; % The two slices to be plotted
    % Create a tiled layout for the plots
    figure;
    t = tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(t, 'Noise Level 10,20,30-Voxel Squared Error Difference', 'FontSize', 24, 'FontWeight', 'bold');

    % Loop over each noise level
    for i = 1:length(eValues)
       E = eValues{i}; % Get the voxel error difference for this noise level
       for j=1:length(sliceNumbers)
            errorSlice = E(:, :, sliceNumbers(j)); % Extract the specified slice
            nexttile;
            titleText = sprintf('Slice:%d', sliceNumbers(j));
            plotErrorSlice(errorSlice, titleText, fontSize, tickFontSize, xLabelText, yLabelText);
       end
 
    end
end


function plotErrorSlice(errorSlice, titleText, fontSize, tickFontSize, xLabelText, yLabelText)
    imagesc(errorSlice);
    colormap(jet); % Use a colormap for visualization
    colorbar;      % Add a colorbar to indicate intensity values
    title(titleText, ...
        'FontSize', fontSize, 'FontWeight', 'bold');
    xlabel(xLabelText, 'FontSize', fontSize);
    ylabel(yLabelText, 'FontSize', fontSize);
    ax = gca; % Get the current axes handle
    set(ax, 'FontSize', tickFontSize); % Set font size for ticks
    axis image; % Ensure the aspect ratio is correct
    clim([0, max(errorSlice(:))]); % Set color limits based on slice values
end

%% mse and plot ----
function mseValues = getRMSE(origVol,....
    gaussianDenoisedVols, ...
    medianFilterDenoisedVols, ...
    bilateralFilterDenoisedVols, ...
    peronaMalikFilterDenoisedVols,...
    geometryPreservingAnisotropicDiffFilterDenoisedVols)
% origVol: the original non-noisy image volume
% gaussianDenoisedVols, medianFilterDenoisedVols, ...
% bilateralFilterDenoisedVols, peronaMalikFilterDenoisedVols,
% geometryPreservingAnisotropicDiffFilterDenoisedVols: different
% denoised volumes of the orignal volume using five filters:
% Gaussian, Median, Bilateral, Peron-Malik Filter and
% Geometry Preserving Anisotropic Diffusion Filter.

numNoiseLevels = size(gaussianDenoisedVols, 2);
mseValues = zeros(numNoiseLevels, 5);

% Iterate over each noise level
for n = 1:numNoiseLevels
   % Calculate the Mean Square Error (MSE)
   smoothedI = gaussianDenoisedVols{n};
   mseValues(n, 1) = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   smoothedI = medianFilterDenoisedVols{n};
   mseValues(n, 2) = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   smoothedI = bilateralFilterDenoisedVols{n};
   mseValues(n, 3) = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   smoothedI = peronaMalikFilterDenoisedVols{n};
   mseValues(n, 4) = sqrt(mean((smoothedI(:) - origVol(:)).^2));
   smoothedI = geometryPreservingAnisotropicDiffFilterDenoisedVols{n};
   mseValues(n, 5) = sqrt(mean((smoothedI(:) - origVol(:)).^2));
end

end

function plotMSE(mseValues)

    % Initialize variables    
    noiseLevels = categorical({'10', '20', '30'});
    
    % Plot the MSE values
    figure('Position', [100, 100, 800, 600]);

    hold on;
    markers = {'-o', '-s', '-^', '-d', '-v'};  % Markers for five filters
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
    filterNames = {'Gaussian', 'Median', 'Bilateral',...
        'Perona-Malik', 'Geometry Preserving Anisotropic Diffusion Filter'};
    legend(filterNames, 'Location', 'best'); 
    grid on;
end
%%
function bestParams = gridSearch(vol)
    % Define parameter ranges for grid search for GeometryPreservingAnisotropicDiffFilter
    sigmaSs = [1, 2, 3];     % Range for sigmaS
    sigmaGs = [0.5, 1, 1.5]; % Range for sigmaG
    Ts = [5, 10, 15];        % Range for number of iterations T
    dts = [0.1, 0.2, 0.3];   % Range for time increment dt

    % Initialize variables to store the best parameters and MSE
    bestMSE = inf;            % Best MSE for this filter
    bestParams = [];          % Store the best parameters for the filter

    % Loop over each noise level
    for noiseIdx = 1:3
        % Generate noisy image
        noisyImage = applyNoise(vol, noiseIdx * 10);
        
        % GeometryPreservingAnisotropicDiffFilter Grid Search
        for sigmaS = sigmaSs
            for sigmaG = sigmaGs
                for T = Ts
                    for dt = dts
                        % Apply GeometryPreservingAnisotropicDiffFilter
                        denoisedVol = computeGeometryPreservingAnisotropicDiffFilter(noisyImage, T, dt, sigmaS, sigmaG, 0.5, 0.9);
                        
                        % Calculate the MSE between denoised and original volume
                        mse = calculateRMSE(vol, denoisedVol);
                        
                        % Update best parameters if current MSE is better
                        if mse < bestMSE
                            bestMSE = mse;
                            bestParams = [sigmaS, sigmaG, T, dt];
                        end
                    end
                end
            end
        end
    end

    % Display the best parameters for GeometryPreservingAnisotropicDiffFilter
    disp('Optimal Parameters for GeometryPreservingAnisotropicDiffFilter:');
    disp(['sigmaS = ', num2str(bestParams(1)), ...
          ', SigmaG = ', num2str(bestParams(2)), ...
          ', T = ', num2str(bestParams(3)), ...
          ', dt = ', num2str(bestParams(4)), ...
          ', MSE = ', num2str(bestMSE)]);
end

%% Function to calculate MSE
function mse = calculateRMSE(origVol, denoisedVol)
    mse = sqrt(mean((origVol(:) - denoisedVol(:)).^2));
end
%%
function saveImages(data, filename)
for i = 1:3
    img = data{i};
    % Define the filename for the NIfTI file
    niftiFilename = sprintf(filename, i);
    % Save as a NIFTI file
    niftiwrite(img, niftiFilename);
end   
end
function data = loadImages(filename)
data = {};
for i = 1:3
    niftiFilename = sprintf(filename, i);
    data{i} = niftiread(niftiFilename);
end
end