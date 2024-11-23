
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

% Compute the histogram
[counts, x] = imhist(slice);

%T = otsuthresh(counts);
T = otsuThreshold(counts);
BW = imbinarize(covslice,T);
showPreprocessingImages(slice, BW, T)


%%
function [mu0, mu1, N] = makeMeanTables(h, K)
    % Compute mean tables (background and foreground) for Otsu's method
    
    % Initialize variables for background (mu0)
    n0 = 0; s0 = 0;
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
    n1 = 0; s1 = 0;
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
    % OTSU's Thresholding Algorithm
    % Input: h - grayscale histogram (vector)
    % Output: threshold - optimal threshold value (or -1 if no threshold is found)

    K = length(h); % Number of intensity levels
    [mu0, mu1, N] = makeMeanTables(h, K); % Compute mean tables for background and foreground
    
    % Initialize variables
    sigma2_b_max = 0; % Maximum between-class variance
    q_max = -1; % Optimal threshold index
    n0 = 0; % Initialize class 0 pixel count
    
    % Loop through all possible thresholds
    for q = 1:K-2
        n0 = n0 + h(q); % Update class 0 pixel count
        n1 = N - n0;    % Class 1 pixel count
        if n0 > 0 && n1 > 0
            % Compute between-class variance
            sigma2_b = (1 / N^2) * n0 * n1 * (mu0(q) - mu1(q))^2;
            if sigma2_b > sigma2_b_max
                sigma2_b_max = sigma2_b;
                q_max = q;
            end
        end
    end

    % Return the optimal threshold
    threshold = q_max - 1; 
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
function showPreprocessingImages(Img, binaryImg, threshold)
figure;
t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Add a title for the entire figure
txt = sprintf('Slice 143 original (left) binary (right) with minimum error thresholding:%d', threshold);
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