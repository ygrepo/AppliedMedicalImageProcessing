clearvars
clc
vol = niftiread('data/sub-11_T1w.nii.gz');
vol = flip (permute(vol, [2 1 3]), 1);
%%
% Generate noisy images ----
noisyImages = {};
noisyImages{1} = applyNoise(vol, 10);
noisyImages{2} = applyNoise(vol, 20);
noisyImages{3} = applyNoise(vol, 30);
%showNoisyImages(noisyImages);
%%
clc
sigmaD = 2;
T = 10;
I = noisyImages{1};
slice = zeros(size(I,1), size(I,2),1);
slice(:,:,1) =I(:, :, 1);
slice = GeometryPreservingAnisotropicDiffFilter(slice,...
    T, sigmaD);  % Apply  filter
disp('done')

% filteredI = computeGeometryPreservingAnisotropicDiffFilter(noisyImages{1},...
%     T, sigmaD);
%% GeometryPreservingAnisotropicDiffFilterilter ----
function filteredI = computeGeometryPreservingAnisotropicDiffFilter(I,...
    T, sigmaD)

filteredI = zeros(size(I));  % Initialize an array to store the filtered image
[~, ~, dim3] = size(I); 
 for i = 1:dim3
    slice = zeros(size(I,1), size(I,2),1);
    slice(:,:,1) =I(:, :, i);
    slice = GeometryPreservingAnisotropicDiffFilter(slice,...
        T, sigmaD);  % Apply  filter
    break;
    filteredI(:, :, i) = slice;  % Store the filtered slice back
 end
end

function I = GeometryPreservingAnisotropicDiffFilter(I, T, sigmaD)
    % I: input image of size M x N with K channels
    % T: number of iterations
    % dt: time increment
    % sigma_g: width of Gaussian for smoothing the gradient
    % sigma_s: width of Gaussian for smoothing the structure matrix
    % sigma_d: width of Gaussian for gradient component smoothing
    % a1, a2: diffusion parameters for directions of min/max variation
    
    % Define the size of the image
    [M, N, K] = size(I);
    
    % Initialize auxiliary matrices
    D = zeros(K, M, N, 2);          % Gradient maps for each channel
    H = zeros(K, M, N, 2, 2);       % Structure matrix
    G = zeros(M, N, 2, 2);          % Combined gradient map
    A = zeros(M, N, 2, 2);          % Diffusion matrix
    B = zeros(K, M, N);             % Update map
    
% Iterate for T steps
% Iterate for T steps
for t = 1:T
    % Initialize structure matrix G for this iteration
    G = zeros(M, N, 2, 2);
    
    % Compute gradient maps D and structure matrix H for each channel
    for k = 1:K
        % Compute the gradient maps for the entire channel k at once
        [Ix_smoothed, Iy_smoothed] = computeGradient(I(:, :, k), sigmaD);
        
        % Store the gradients in D for the current channel
        D(k, :, :, 1) = Ix_smoothed;  % Gradient in x-direction
        D(k, :, :, 2) = Iy_smoothed;  % Gradient in y-direction
        
        % Compute the Hessian matrix H for the entire channel k at once
        H_k = computeHessian(I(:, :, k));  % Call with the k-th channel slice
        
        % Store the Hessian matrix for channel k in the overall H matrix
        H(k, :, :, :, :) = H_k;
        
        % Calculate the structure matrix G by accumulating the contributions from each channel
        G(:, :, 1, 1) = G(:, :, 1, 1) + Ix_smoothed .^ 2;              % G0 = sum_k (Ix_smoothed^2)
        G(:, :, 1, 2) = G(:, :, 1, 2) + Ix_smoothed .* Iy_smoothed;     % G1 = sum_k (Ix_smoothed * Iy_smoothed)
        G(:, :, 2, 1) = G(:, :, 1, 2);                                 % G1 (symmetric)
        G(:, :, 2, 2) = G(:, :, 2, 2) + Iy_smoothed .^ 2;              % G2 = sum_k (Iy_smoothed^2)
    end
end
end

function I = GeometryPreservingAnisotropicDiffFilter2(I, T, dt, sigma_g, sigma_s, a1, a2)
    % I: input image of size M x N with K channels
    % T: number of iterations
    % dt: time increment
    % sigma_g: width of Gaussian for smoothing the gradient
    % sigma_s: width of Gaussian for smoothing the structure matrix
    % a1, a2: diffusion parameters for directions of min/max variation
    
    % Define the size of the image
    [M, N, K] = size(I);
    
    % Initialize auxiliary matrices
    D = zeros(K, M, N, 2);          % Gradient maps for each channel
    H = zeros(K, M, N, 2, 2);       % Structure matrix
    G = zeros(M, N, 2, 2);          % Combined gradient map
    A = zeros(M, N, 2, 2);          % Diffusion matrix
    B = zeros(K, M, N);             % Update map
    
    % Iterate for T steps
    for t = 1:T
        % Compute gradient maps D and structure matrix H for each channel
        for k = 1:K
            for u = 1:M
                for v = 1:N
                    % Compute gradients for channel k
                    D(k, u, v, :) = computeGradient(I(:, :, k), u, v);                    
               end
            end
        end
        
        % Smooth structure matrices H across all channels
        H_smooth = gaussianSmooth(H, sigma_s);
        
        % Compute combined gradient map G
        for u = 1:M
            for v = 1:N
                G(u, v, :, :) = sum(squeeze(H_smooth(:, u, v, :, :)), 1);
            end
        end
        
        % Smooth combined gradient map G
        G_smooth = gaussianSmooth(G, sigma_s);
        
        % Compute eigenvalues and eigenvectors for G and populate A
        for u = 1:M
            for v = 1:N
                % Extract matrix G(u, v)
                Guv = squeeze(G_smooth(u, v, :, :));
                
                % Compute eigenvalues and eigenvectors
                [V, L] = eig(Guv);
                lambda1 = L(1, 1);
                lambda2 = L(2, 2);
                
                e1 = V(:, 1) / norm(V(:, 1));
                x1 = e1(1); y1 = e1(2);
                
                % Diffusion coefficients
                c1 = 1 / (1 + lambda1^2) ^ a1;
                c2 = 1 / (1 + lambda2^2) ^ a2;
                
                % Populate A(u, v)
                A(u, v, :, :) = [c1 * y1^2 + c2 * x1^2, (c2 - c1) * x1 * y1; ...
                                 (c2 - c1) * x1 * y1, c1 * x1^2 + c2 * y1^2];
            end
        end
        
        % Compute B matrix for each channel and find beta_max
        beta_max = -inf;
        for k = 1:K
            for u = 1:M
                for v = 1:N
                    B(k, u, v) = trace(squeeze(A(u, v, :, :)) * squeeze(H_smooth(k, u, v, :, :)));
                    beta_max = max(beta_max, abs(B(k, u, v)));
                end
            end
        end
        
        % Set alpha parameter
        alpha = dt / beta_max;
        
        % Update I based on B matrix
        for k = 1:K
            for u = 1:M
                for v = 1:N
                    I(u, v, k) = I(u, v, k) + alpha * B(k, u, v);
                end
            end
        end
    end
end


function [Ix_smoothed, Iy_smoothed] = computeGradient(I, sigmaD)
    % Compute the gradients across the whole image
    a = (2 - sqrt(2)) / 4;
    b = (sqrt(2) - 1) / 2;

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
    Ix_smoothed = imgaussfilt(Ix, sigmaD);
    Iy_smoothed = imgaussfilt(Iy, sigmaD);
end

function H = computeHessian(I_channel)
    % computeHessian Computes the Hessian matrix for a 2D image or image channel
    % I_channel: 2D input image for a single channel (MxN)
    % H: 4D output Hessian matrix of size (M x N x 2 x 2)
    %    H(u, v, 1, 1) = ∂²I/∂x² at pixel (u, v)
    %    H(u, v, 1, 2) = ∂²I/∂x∂y at pixel (u, v)
    %    H(u, v, 2, 1) = ∂²I/∂x∂y at pixel (u, v)
    %    H(u, v, 2, 2) = ∂²I/∂y² at pixel (u, v)
    
    % Define the Hessian kernels based on the given specifications
    Hxx_kernel = [1, -2, 1];          % 1x3 kernel for second derivative in x
    Hyy_kernel = [1; -2; 1];          % 3x1 kernel for second derivative in y
    Hxy_kernel = [1/4, 0, -1/4;       % 3x3 kernel for mixed second derivative
                  0,   0,  0;
                 -1/4, 0,  1/4];
    
    % Compute the second-order partial derivatives using convolution
    Hxx = conv2(I_channel, Hxx_kernel, 'same');  % ∂²I / ∂x²
    Hyy = conv2(I_channel, Hyy_kernel, 'same');  % ∂²I / ∂y²
    Hxy = conv2(I_channel, Hxy_kernel, 'same');  % ∂²I / ∂x∂y (mixed derivative)
    
    % Initialize the output Hessian matrix H
    [M, N] = size(I_channel);
    H = zeros(M, N, 2, 2);
    
    % Populate the Hessian matrix
    H(:, :, 1, 1) = Hxx;  % Hxx (∂²I / ∂x²)
    H(:, :, 1, 2) = Hxy;  % Hxy (∂²I / ∂x∂y)
    H(:, :, 2, 1) = Hxy;  % Hxy (∂²I / ∂x∂y)
    H(:, :, 2, 2) = Hyy;  % Hyy (∂²I / ∂y²)
end


function G = computeStructureMatrix(Dx, Dy)
    % computeStructureMatrix Computes the structure matrix G for a multi-channel image
    % Dx: 3D array of smoothed gradients in the x-direction (K x M x N)
    % Dy: 3D array of smoothed gradients in the y-direction (K x M x N)
    % G: 4D structure matrix of size (M x N x 2 x 2)
    %    G(:, :, 1, 1): G0 = sum_k (Dx_k^2)
    %    G(:, :, 1, 2): G1 = sum_k (Dx_k * Dy_k)
    %    G(:, :, 2, 1): G1 (symmetric)
    %    G(:, :, 2, 2): G2 = sum_k (Dy_k^2)
    
    % Get the dimensions
    [K, M, N] = size(Dx);  % K: number of channels, MxN: image size
    
    % Initialize the structure matrix G
    G = zeros(M, N, 2, 2);
    
    % Loop over each channel and accumulate the structure matrix components
    for k = 1:K
        % Accumulate the components of G for each pixel location
        G(:, :, 1, 1) = G(:, :, 1, 1) + Dx(k, :, :) .^ 2;          % G0 = sum_k (Dx_k^2)
        G(:, :, 1, 2) = G(:, :, 1, 2) + Dx(k, :, :) .* Dy(k, :, :); % G1 = sum_k (Dx_k * Dy_k)
        G(:, :, 2, 1) = G(:, :, 1, 2);                             % G1 (symmetric)
        G(:, :, 2, 2) = G(:, :, 2, 2) + Dy(k, :, :) .^ 2;          % G2 = sum_k (Dy_k^2)
    end
end


% function gradient = computeGradient(I, u, v, sigmaD)
%     % Computes the smoothed gradient at pixel (u, v) using convolution with H_x and H_y
%     % and then applies Gaussian smoothing
% 
%     % Constants based on the given formulas
%     a = (2 - sqrt(2)) / 4;
%     b = (sqrt(2) - 1) / 2;
% 
%     % Define convolution kernels H_x and H_y
%     Hx = [-a, 0, a;
%           -b, 0, b;
%           -a, 0, a];
% 
%     Hy = [-a, -b, -a;
%            0,  0,  0;
%            a,  b,  a];
% 
%     % Perform custom convolution to compute the gradients in x and y directions
%     Ix = conv2(I, Hx, 'same');
%     Iy = conv2(I, Hy, 'same');
%     gradient = [Ix(u, v), Iy(u, v)];
%     % Apply Gaussian smoothing to the gradient components
%     % Ix_smoothed = gaussianSmooth(Ix, sigmaD);
%     % Iy_smoothed = gaussianSmooth(Iy, sigmaD);
% 
%     % Output the smoothed gradient vector at location (u, v)
%     %gradient = [Ix_smoothed(u, v), Iy_smoothed(u, v)];
% end

function output = customConv2(I, kernel)
    % Custom 2D convolution function
    % I: input image (M x N)
    % kernel: convolution kernel (m x n)
    % output: convolved image (M x N), same size as input
    
    [M, N] = size(I);             % Size of the input image
    [m, n] = size(kernel);        % Size of the kernel
    
    % Calculate padding size
    pad_m = floor(m / 2);
    pad_n = floor(n / 2);
    
    % Pad the input image with zeros around the border
    I_padded = padarray(I, [pad_m, pad_n], 0, 'both');
    
    % Initialize output matrix
    output = zeros(M, N);
    
    % Perform convolution
    for i = 1:M
        for j = 1:N
            % Extract the region of interest from the padded image
            region = I_padded(i:i+m-1, j:j+n-1);
            
            % Perform element-wise multiplication and sum the result
            output(i, j) = sum(sum(region .* kernel));
        end
    end
end

function smoothed = gaussianSmooth(I, sigmaD)
    % Applies isotropic Gaussian smoothing with standard deviation sigma_d
    % I: input image (M x N) - gradient image (I_x or I_y)
    % sigma_d: standard deviation for Gaussian filter
    % smoothed: output image (M x N), same size as input

    % Apply Gaussian filtering using MATLAB's built-in imgaussfilt function
    smoothed = imgaussfilt(I, sigmaD);
end


% function smoothed = gaussianSmooth(I, sigma_d)
%     % Applies isotropic Gaussian smoothing with standard deviation sigma_d
%     % I: input image (M x N) - gradient image (I_x or I_y)
%     % sigma_d: standard deviation for Gaussian kernel
%     % smoothed: output image (M x N), same size as input
% 
%     % Determine kernel size based on sigma_d
%     kernel_size = 2 * ceil(3 * sigma_d) + 1;  % Rule of thumb for Gaussian kernel size
% 
%     % Generate Gaussian kernel
%     G = fspecialGaussian(kernel_size, sigma_d);
% 
%     % Apply custom convolution with Gaussian kernel
%     smoothed = customConv2(I, G);
% end
% 
% function G = fspecialGaussian(size, sigma)
%     % Generates a Gaussian kernel of a given size and standard deviation sigma
%     % size: size of the kernel (e.g., 5x5, 7x7)
%     % sigma: standard deviation of the Gaussian
%     % G: output Gaussian kernel
% 
%     % Create a grid of (x, y) coordinates
%     [x, y] = meshgrid(-floor(size/2):floor(size/2), -floor(size/2):floor(size/2));
% 
%     % Gaussian function
%     G = exp(-(x.^2 + y.^2) / (2 * sigma^2));
% 
%     % Normalize to ensure the sum is 1
%     G = G / sum(G(:));
% end

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