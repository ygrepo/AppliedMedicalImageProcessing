% Example 3x3 intensity matrix (image patch)
%I = [100, 150, 200; 120, 180, 220; 130, 140, 210];
I = [1, 2; 4 5];

% Dimensions of the matrix
[M, N] = size(I);

% Step 1: Calculate the mean intensity (I_bar)
I_bar = (1 / (M * N)) * sum(I(:));

% Step 2: Calculate the RMS contrast
Crms = sqrt( (1 / (M * N - 1)) * sum((I(:) - I_bar).^2) );

% Display the results
fprintf('Mean Intensity (I_bar): %f\n', I_bar);
fprintf('RMS Contrast (Crms): %f\n', Crms);

K = 8;
h = computeHisto(K, I(:), 2)
%[N, edges, bin] = histcounts(I(:), 2);
h_norm = h / sum(h);
% Entropy = - sum(h_norm(j) * log2(h_norm(j)))
entrop = -sum(h_norm .* log2(h_norm))
%%
entropy(I)
%%
% Define the image patch
I = [0, 1, 2; 1, 2, 3; 2, 3, 3];

% Step 1: Calculate the histogram (number of occurrences of each intensity value)
% Assuming intensity values range from 0 to 3 (4 possible intensity values)
numLevels = 4;  % Intensity levels: 0, 1, 2, 3
histogram = zeros(1, numLevels);  % Initialize histogram

% Manually compute the histogram
for intensity = 0:(numLevels-1)
    histogram(intensity + 1) = sum(I(:) == intensity);  % Count occurrences of each intensity
end

% Display the histogram
disp('Histogram:');
disp(histogram);

% Step 2: Normalize the histogram (to get the probability distribution)
totalPixels = numel(I);  % Total number of pixels
normalized_histogram = histogram / totalPixels;

% Display the normalized histogram
disp('Normalized Histogram (Probability Distribution):');
disp(normalized_histogram);

% Step 3: Compute the CDF (Cumulative Distribution Function)
cdf = cumsum(histogram);  % Cumulative sum of the normalized histogram

% Display the CDF
disp('CDF (Cumulative Distribution Function):');
disp(cdf);
