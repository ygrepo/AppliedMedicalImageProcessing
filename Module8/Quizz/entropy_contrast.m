% Example intensity array (1D or 2D)
I = [23, 56, 78, 23, 56, 23, 78, 90, 45, 56, 78, 90]; % Example 1D intensity array
% You can replace this with a grayscale image matrix as well, e.g. I = imread('image.png');

% Normalize the intensity values if the image is not already in the range [0, K-1]
% Here we assume a bit-depth of 8 bits, so K = 256
I = double(I); % Convert to double precision
K = 256; % Total possible intensity values (0-255 for 8-bit image)

% Calculate histogram of intensity values
%h = histcounts(I(:), 0:K); % This calculates the number of occurrences of each intensity value
h = computeHisto(K, I(:));

% Normalize the histogram to get probabilities h(j)_norm
h_norm = h / sum(h); % Total sum should be 1 (probability distribution)

% Avoid log2(0) issue by replacing 0 probabilities with a small value (log2(0) is undefined)
h_norm(h_norm == 0) = eps; % eps is a small constant in MATLAB

% Calculate the entropy using the formula:
% Entropy = - sum(h_norm(j) * log2(h_norm(j)))
entrop = -sum(h_norm .* log2(h_norm));

% Display the result
fprintf('Entropy Contrast: %f\n', entrop);
