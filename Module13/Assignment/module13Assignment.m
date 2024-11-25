
clearvars
clc
vol = niftiread('data/sub-13_T1w.nii.gz');
%vol = flip (permute(vol, [2 3 1]), 1);
vol = flipdim(permute (vol, [2 1 3]) ,1);
%%
%%volshow(vol)
volumeViewer(vol)
%% Look at Slice 102 and Slice 119 ---
sliceNumber = 143;
% Display slice 102 using imshow
figure;
imshow(vol(:,:,sliceNumber), []);
title('Slice 143');
hold off;
%%
slice = vol(:,:,sliceNumber);
h = imhist(slice); % Compute histogram
T = minimumErrorThreshold(h);
binaryImage = slice > T;
disp(['Optimal Threshold: ', num2str(T)]);
showPreprocessingImages(slice, binaryImage, T)

% Normalize the slice to the range [0, 1] if it contains floating-point values
if ~isa(slice, 'uint8') && ~isa(slice, 'uint16') 
    covslice = mat2gray(slice); % Normalize to [0, 1]
end

% Compute the histogram
h= imhist(covslice);
% Compute the normalized threshold
T_normalized = T / (length(h) - 1); % Normalize raw threshold to [0, 1]

eta = computeAbsoluteGoodness(h, T_normalized); % Compute absolute goodness
disp(['Absolute Goodness (η): ', num2str(eta)]);

%%
slice = vol(:,:,sliceNumber); % Extract the specific slice

% Normalize the slice to the range [0, 1] if it contains floating-point values
if ~isa(slice, 'uint8') && ~isa(slice, 'uint16') % Check if the image is not already in integer format
    covslice = mat2gray(slice); % Normalize to [0, 1]
end

% Compute the histogram
[counts, x] = imhist(covslice);

% Plot the histogram
figure;
stem(x, counts, 'Marker', 'o', 'LineWidth', 1.5); % Use 'stem' for discrete histogram
xlabel('Intensity Value');
ylabel('Pixel Count');
title('Histogram of Slice');
grid on; 

%%
% Assume 'vol' is your 3D volume and 'sliceNumber' is the slice index
slice = vol(:,:,sliceNumber); % Extract the specific slice

% Normalize the slice to the range [0, 1] if it contains floating-point values
if ~isa(slice, 'uint8') && ~isa(slice, 'uint16') 
    covslice = mat2gray(slice); % Normalize to [0, 1]
end

% Compute the histogram
h= imhist(covslice);

T = otsuthresh(h);
eta = computeAbsoluteGoodness(h, T); % Compute absolute goodness
disp(['Absolute Goodness (η): ', num2str(eta)]);
BW = imbinarize(covslice,T);
showPreprocessingImages(slice, BW, T);

%%
% Assume 'vol' is your 3D volume and 'sliceNumber' is the slice index
slice = vol(:,:,sliceNumber); % Extract the specific slice

% Normalize the slice to the range [0, 1] if it contains floating-point values
if ~isa(slice, 'uint8') && ~isa(slice, 'uint16') 
    covslice = mat2gray(slice); % Normalize to [0, 1]
end

% Compute the histogram
h= imhist(covslice);

T = otsuThreshold(h);
eta = computeAbsoluteGoodness(h, T); % Compute absolute goodness
disp(['Absolute Goodness (η): ', num2str(eta)]);
BW = imbinarize(covslice,T);
showPreprocessingImages(slice, BW, T);
%%
K = 5; % Number of tissue classes
beta = 0.5; % Smoothness weight
maxIter = 10; % Maximum number of iterations
order = 2; % Polynomial order for the gain field
tol = 1e-3; % Convergence tolerance
[slice,z] = unsupervisedClassification(vol,sliceNumber,...
    K, beta, maxIter, order, tol);

%%
showProcessedImage(slice, z, K)

%%
function [mu0, mu1, N] = makeMeanTables(h, K)
    % Compute mean tables (background and foreground) for Otsu's method
    
    % Initialize variables for background (mu0)
    n0 = 0; 
    s0 = 0;
    mu0 = zeros(1, K);
    for q = 1:K
        n0 = n0 + h(q);
        s0 = s0 + (q - 1) * h(q);
        if n0 > 0
            mu0(q) = s0 / n0; % Mean for class 0
        else
            mu0(q) = -1; % Invalid mean
        end
    end
    
    % Initialize variables for foreground (mu1)
    N = n0; % Total number of pixels
    n1 = 0; 
    s1 = 0;
    mu1 = zeros(1, K);
    for q = K:-1:2
        n1 = n1 + h(q);
        s1 = s1 + (q - 1) * h(q);
        if n1 > 0
            mu1(q - 1) = s1 / n1; % Mean for class 1
        else
            mu1(q - 1) = -1; % Invalid mean
        end
    end
end

function threshold = otsuThreshold(h)
    % OTSU's Thresholding Algorithm using makeMeanTables
    % Input: h - grayscale histogram (vector)
    % Output: 
    %   threshold - optimal threshold value (normalized to [0, 1])
    %   effectiveness - effectiveness metric for the threshold

    K = length(h); % Number of intensity levels

    % Compute mean tables (mu0 for background, mu1 for foreground)
    [mu0, mu1, N] = makeMeanTables(h, K); % Helper function

    % Initialize variables
    sigma2_b_max = 0; % Maximum between-class variance
    q_max = -1; % Optimal threshold index

    % Loop through all possible thresholds
    for q = 1:K-1
        % Compute the cumulative probabilities for background and foreground
        n0 = sum(h(1:q)); % Class 0 pixel count
        n1 = N - n0;      % Class 1 pixel count

        if n0 > 0 && n1 > 0
            % Compute between-class variance
            sigma2_b = (1 / N^2) * n0 * n1 * (mu0(q) - mu1(q))^2;
            if sigma2_b > sigma2_b_max
                sigma2_b_max = sigma2_b;
                q_max = q;
            end
        end
    end

    % Return the optimal threshold (normalized to [0, 1])
    if q_max == -1
        threshold = 0; % Default to 0 if no threshold is found
    else
        threshold = (q_max - 1) / (K - 1); % Normalize threshold
    end

end


function eta = computeAbsoluteGoodness(h, T)
    % Compute "absolute goodness" metric for Otsu's thresholding 
    
    % Inputs:
    % h - Grayscale histogram (vector)
    % T - Threshold (normalized or scaled, will be converted to index)
    
    % Total number of pixels
    N = sum(h);
    
    % Compute intensity levels corresponding to histogram bins
    indices = 0:length(h)-1; % Intensity levels
    
    % Compute total variance (sigma_I^2) using MATLAB's var
    sigma_I2 = var(repelem(indices, h')); % Weighted variance of intensities
    
    % Convert normalized threshold T to histogram index
    q_max = round(T * (length(h) - 1)); % Scale normalized threshold to bin index
    
    % Split the histogram into two classes
    class0_indices = indices(1:q_max); % Background (class 0)
    class1_indices = indices(q_max+1:end); % Foreground (class 1)
    
    % Compute pixel counts for each class
    n0 = sum(h(1:q_max)); % Class 0 pixel count
    n1 = sum(h(q_max+1:end)); % Class 1 pixel count
    
    % Ensure n0 and n1 are valid
    if n0 > 0 && n1 > 0
        % Compute means for each class (weighted mean using MATLAB's mean)
        mu0 = mean(repelem(class0_indices, h(1:q_max)')); % Mean of class 0
        mu1 = mean(repelem(class1_indices, h(q_max+1:end)')); % Mean of class 1
        
        % Compute between-class variance
        sigma_b2 = (n0 * n1 / N^2) * (mu0 - mu1)^2; % Scalar subtraction
    else
        sigma_b2 = 0; % No between-class variance
    end
    
    % Compute absolute goodness (eta)
    eta = sigma_b2 / sigma_I2;
end


function [sigma2_0, sigma2_1, N] = makeSigmaTable(h, K)
    % Initialize variables
    n0 = 0;
    A0 = 0;
    B0 = 0;
    sigma2_0 = zeros(1, K);
    
    % Calculate sigma2_0 for each q
    for q = 1:K
        n0 = n0 + h(q);
        A0 = A0 + h(q) * (q - 1);
        B0 = B0 + h(q) * (q - 1)^2;
        if n0 > 0
            sigma2_0(q) = 1/12 + (B0 - (A0^2 / n0)) / n0;
        else
            sigma2_0(q) = 0;
        end
    end
    
    % Remaining variables for sigma2_1
    N = n0; % Total pixel count
    n1 = 0;
    A1 = 0;
    B1 = 0;
    sigma2_1 = zeros(1, K);
    
    for q = K-1:-1:1
        n1 = n1 + h(q + 1);
        A1 = A1 + h(q + 1) * (q);
        B1 = B1 + h(q + 1) * (q)^2;
        if n1 > 0
            sigma2_1(q) = 1/12 + (B1 - (A1^2 / n1)) / n1;
        else
            sigma2_1(q) = 0;
        end
    end
end

function t = minimumErrorThreshold(h)
    % Input: h - grayscale histogram
    % Output: t - optimal threshold for binary classification
    
    K = length(h); % Number of intensity levels
    [sigma2_0, sigma2_1, N] = makeSigmaTable(h, K); % Get sigma tables and total pixel count
    
    % Initialize variables
    n0 = 0;
    q_min = -1;
    e_min = Inf;
    
    % Iterate through all possible thresholds
    for q = 1:K-2
        n0 = n0 + h(q);
        n1 = N - n0;
        if n0 > 0 && n1 > 0
            P0 = n0 / N; % Probability of class 0
            P1 = n1 / N; % Probability of class 1
            
            % Compute the error
            e = P0 * log(sigma2_0(q)) + P1 * log(sigma2_1(q)) - ...
                2 * (P0 * log(P0) + P1 * log(P1));
            
            % Update minimum error and threshold
            if e < e_min
                e_min = e;
                q_min = q;
            end
        end
    end
    
    % Return the optimal threshold
    t = q_min - 1; 
end

%%
function centroids = initializeCentroids(img, K)
    % Initialize K centroids evenly spaced based on the image intensity range
    %
    % Inputs:
    % img - Input grayscale image (2D array)
    % K   - Number of tissue classes (centroids to initialize)
    %
    % Outputs:
    % centroids - A 1xK vector of evenly spaced centroids

    % Convert the image to double for computations
    img = double(img);
    
    % Compute the intensity range of the image
    minIntensity = min(img(:));
    maxIntensity = max(img(:));
    
    % Evenly space centroids across the intensity range
    centroids = linspace(minIntensity, maxIntensity, K);
end


function z = estimateIndicatorFunctionWithGain_ICM_V(img, gamma, g, beta, K, maxIter)
    % Estimate the indicator function z using ICM with V matrix
    % Inputs:
    % img     - Input grayscale image (2D array)
    % gamma   - Class means (centroids) [1 x K]
    % g       - Gain field (rows*cols x 1 vector)
    % beta    - Weight for smoothness term
    % K       - Number of classes
    % maxIter - Maximum number of iterations
    %
    % Output:
    % z - Indicator function (rows*cols x K matrix)

    % Image dimensions
    [rows, cols] = size(img);
    img = double(img); % Ensure the image is in double format
    N = rows * cols; % Total number of pixels

    % Flatten the image for convenience
    imgFlat = img(:);

    % Ensure g is flattened
    g = g(:);

    % Initialize z randomly (one-hot encoded)
    z = zeros(N, K);
    for i = 1:N
        z(i, randi(K)) = 1; % Random initial assignment
    end

    % Precompute spatial neighbors
    neighbors = getNeighbors(rows, cols);

    % Precompute the V matrix
    V = ones(K, K) - eye(K); % V matrix: 1-penalty for different classes, 0 for same class

    % ICM Iteration
    for iter = 1:maxIter
        fprintf('Iteration %d/%d\n', iter, maxIter);

        % Loop through each pixel
        for j = 1:N
            % Compute the energy for each class
            energies = zeros(1, K);
            for k = 1:K
                % Data term: (y_j - g_j * gamma_k)^2
                dataTerm = (imgFlat(j) - g(j) * gamma(k))^2;

                % Smoothness term: beta * sum_neighbors z_j^T V z_i
                smoothnessTerm = 0;
                for neighbor = neighbors{j}'
                    smoothnessTerm = smoothnessTerm + z(neighbor, :) * V * z(j, :)';
                end

                % Total energy for class k
                energies(k) = dataTerm + beta * smoothnessTerm;
            end

            % Assign pixel to the class with minimum energy
            [~, minClass] = min(energies);
            z(j, :) = 0; % Reset the current pixel's class
            z(j, minClass) = 1; % Assign the pixel to the minimum energy class
        end

        % Validate and Normalize z after update
        sumZ = sum(z, 2); % Sum across classes
        if any(abs(sumZ - 1) > 1e-6)
            % Normalize rows to ensure they sum to 1
            z = z ./ sumZ;
        end

        % Check for one-hot encoding consistency
        invalidRows = find(sum(z, 2) ~= 1, 1); % Find rows that do not sum to 1
        if ~isempty(invalidRows)
            error('Invalid z: Some rows are not one-hot encoded after update.');
        end
    end

    % Reshape z for output (rows*cols x K)
    z = reshape(z, [N, K]);
end


function neighbors = getNeighbors(rows, cols)
    % Compute spatial neighbors for each pixel in a 2D image
    % Outputs:
    % neighbors - Cell array where each cell contains the indices of neighbors

    neighbors = cell(rows * cols, 1);
    for r = 1:rows
        for c = 1:cols
            idx = sub2ind([rows, cols], r, c);
            neighborIdx = [];
            if r > 1, neighborIdx = [neighborIdx; sub2ind([rows, cols], r - 1, c)]; end
            if r < rows, neighborIdx = [neighborIdx; sub2ind([rows, cols], r + 1, c)]; end
            if c > 1, neighborIdx = [neighborIdx; sub2ind([rows, cols], r, c - 1)]; end
            if c < cols, neighborIdx = [neighborIdx; sub2ind([rows, cols], r, c + 1)]; end
            neighbors{idx} = neighborIdx;
        end
    end
end

function P = computePolynomialBasis(coords, order)
    % Compute polynomial basis matrix P for coordinates and given order
    %
    % Inputs:
    % coords - Nx2 matrix of (x, y) coordinates
    % order  - Degree of the polynomial basis
    %
    % Output:
    % P - Polynomial basis matrix (N x M)

    x = coords(:, 1);
    y = coords(:, 2);

    % Initialize polynomial basis with [x^0 * y^0]
    P = ones(size(x)); % Start with constant term

    % Add polynomial terms up to the specified order
    for i = 1:order
        for j = 0:i
            P = [P, (x.^(i-j)) .* (y.^j)];
        end
    end
end

function gamma = updateCentroids(z, g, y)
    % Update centroids (gamma_k) using vectorized operations
    % Inputs:
    % z - Indicator function (N x K matrix, one-hot encoded)
    % g - Gain field (N x 1 vector)
    % y - Image intensities (N x 1 vector)
    % K - Number of classes
    %
    % Output:
    % gamma - Updated centroids (1 x K)

    % Numerator: sum_j z_jk * g_j * y_j
    numerator = sum(z .* (g .* y), 1); % (1 x K)

    % Denominator: sum_j z_jk * g_j^2
    denominator = sum(z .* (g.^2), 1); % (1 x K)
    denominator = max(denominator, 1e-8); % Prevent division by zero

    % Compute gamma_k for each class k
    gamma = numerator ./ denominator; % (1 x K)

    % Enforce non-negativity
    gamma = max(gamma, 0); % Gamma should never be negative
end


function [slice,z] = unsupervisedClassification(vol,sliceNumber,...
    K, beta, maxIter, order, tol)
% Parameters
slice = double(vol(:, :, sliceNumber)); % Extract the slice
gamma = initializeCentroids(slice, K); % Initialize centroids
g = ones(size(slice)); % Initialize gain field as flat (all ones)

% Flatten the slice and gain field for processing
sliceFlat = slice(:);
gFlat = g(:);

% Precompute the polynomial basis P (Chebyshev polynomials)
[rows, cols] = size(slice);
[X, Y] = meshgrid(1:cols, 1:rows);
coords = [X(:), Y(:)]; % Flattened coordinates
P = computePolynomialBasis(coords, order); % Polynomial basis matrix

% Iterative algorithm with convergence criteria
prevZ = zeros(size(sliceFlat, 1), K); % Initialize previous z
prevGamma = zeros(1, K); % Initialize previous gamma
prevGFlat = zeros(size(gFlat)); % Initialize previous g

for iter = 1:maxIter
    fprintf('Iteration %d/%d\n', iter, maxIter);

    % Step 1: Update the indicator function (z)
    z = estimateIndicatorFunctionWithGain_ICM_V(slice, gamma, g, beta, K, maxIter);

    % Step 2: Update centroids (gamma)
    gamma = updateCentroids(z, gFlat, sliceFlat);

    % Step 3: Update the gain field (g)
    % Solve for the polynomial coefficients f using least squares
    Y = sliceFlat ./ max(gFlat, 1e-8); % Avoid division by zero
    regParam = 1e-3; % Regularization parameter
    f = (P' * P + regParam * eye(size(P, 2))) \ (P' * Y);

    % Recompute the gain field g from the polynomial coefficients
    gFlat = P * f;

    % Clip g to a reasonable range
    gFlat = max(gFlat, 1e-6); % Ensure g is positive
    g = reshape(gFlat, rows, cols); % Reshape back to 2D

    % Ensure gain field is valid
    if any(isnan(gFlat)) || any(isinf(gFlat))
        error('Gain field contains invalid values.');
    end

    % Compute changes for convergence
    deltaZ = max(abs(z(:) - prevZ(:))); % Maximum change in z
    deltaGamma = max(abs(gamma - prevGamma)); % Maximum change in gamma
    deltaG = max(abs(gFlat - prevGFlat)); % Maximum change in g

    % Display changes (optional)
    fprintf('Max deltaZ: %.6f, Max deltaGamma: %.6f, Max deltaG: %.6f\n', deltaZ, deltaGamma, deltaG);

    % Check for convergence
    if deltaZ < tol && deltaGamma < tol && deltaG < tol
        fprintf('Convergence reached after %d iterations.\n', iter);
        break;
    end

    % Update previous values
    prevZ = z;
    prevGamma = gamma;
    prevGFlat = gFlat;

    % Display intermediate results (optional)
    fprintf('Updated centroids (gamma): ');
    disp(gamma);
end

end

%%
function showPreprocessingImages(Img, binaryImg, threshold)
figure;
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
txt = sprintf('Slice 143 original (left) binary (right) with minimum error thresholding:%5.2f', threshold);
title(t, txt, 'FontSize', 24, 'FontWeight', 'bold');
fontSize = 14;

nexttile;
imshow(Img, []);  % Display original image.
title('Original Image', 'FontSize', fontSize,'FontWeight','bold');
nexttile;
imshow(binaryImg, []);  % Display binary image.
title('Binary Image', 'FontSize', fontSize,'FontWeight','bold');
hold off
end

function showProcessedImage(Img, z, K)
    % Img: Original image
    % z: Indicator function (rows*cols x K matrix)
    % K: Number of tissue classes
    
    % Class labels for segmentation
    classLabels = {'Background', 'White Matter', 'Gray Matter', 'CSF', 'No-Brain Tissue'};
    
    % Generate segmentation from z
    [~, segmentation] = max(z, [], 2); % Assign each pixel to the class with the highest probability
    segmentation = reshape(segmentation, size(Img)); % Reshape to image size

    % Define colormap for segmentation visualization
    cmap = lines(K); % Generate unique colors for each class
    
    % Create tiled layout
    figure;
    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Add a title for the entire figure
    txt = sprintf('Slice 143 original (left), and segmented (right)');
    title(t, txt, 'FontSize', 24, 'FontWeight', 'bold');
    fontSize = 14;

    % Display original image
    nexttile;
    imshow(Img, []); % Display original image
    title('Original Image', 'FontSize', fontSize, 'FontWeight', 'bold');
    
    % Display segmented image
    nexttile;
    imshow(segmentation, []); % Display segmentation
    colormap(cmap); % Apply colormap
    colorbar('Ticks', 1:K, 'TickLabels', classLabels); % Add legend for classes
    title('Segmented Image', 'FontSize', fontSize, 'FontWeight', 'bold');
end


