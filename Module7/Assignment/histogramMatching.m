% Add NIfTI toolbox to the MATLAB path
clearvars
%%
clc
% Specify the directory containing the .nii.gz files
folder = './MRI_T1W';  % Update this path to your directory
trainingImageIndices = [2:7, 9:12];
imageData = loadImages(folder, trainingImageIndices);
% Set up the landmark parameters
imageData = setUpPeaks(imageData);
imageData = setUpPercentiles(imageData);

%%

images = imageData.V(imageData.isTraining);
setIndex = imageData.imageIndex(imageData.isTraining);
plotHistogram(images(1:5),setIndex(1:5));
plotHistogram(images(6:end),setIndex(6:end));

%%

plotHistogram2(imageData.V(imageData.isTraining))

%%
img = imageData.V(imageData.isTraining);
subjectIndex = 10;
img = img{subjectIndex};
landmarks = struct();
flatImg = img(:);
histo = computeHistogram(flatImg, 100);
landmarks.min = min(flatImg);
landmarks.max = max(flatImg); 
landmarks.peaks  = findNthLargestBinsIntensity(histo, [2 6]);
landmarks.percentiles = getPercentiles(flatImg,[10 99.8]); 
subject = imageData.subjectIndex(imageData.isTraining);
subject = subject(subjectIndex);
plotHistogramWithLandmarks(subject, histo, landmarks);

%%
opt = struct();
opt.plotHistogram = false;
opt.numBins = 100;
landmarks = determineLandmarks(imageData, opt);
%%
displayLandmarks(landmarks)
%%
clc
[s1, s2] = finds1s2(landmarks);
%%
applyStandardizations(landmarks{1}, s1, s2)
%standardizedLandmarks = applyStandardizationToLandmarks(landmarks, s1, s2);
%%
clc
displayStandardizedLandmarks(standardizedLandmarks);
%%
clc
combinedAggregatedLandmarks = combineStandardizedLandmarks(standardizedLandmarks);
displayCombinedAggregatedLandmarks(combinedAggregatedLandmarks);
meanLandmarks = computeMeanLandmarks(combinedAggregatedLandmarks);
meanLandmarks
%%
function res = standardize(val, s1, s2, pc1, pc2)
    res = s1 + (val - pc1) * (s2 - s1) / (pc2 - pc1);
end

function result = applyStandardizations(data, s1, s2)
    result = {};
    pc1 = data.percentiles.values(1);
    pc2 = data.percentiles.values(2);
    result.min = standardize(data.min, s1, s2, pc1, pc2);
    result.max = standardize(data.max, s1, s2, pc1, pc2);
    peaks = [];
    for i=1:size(data.peaks,2)
        peaks(i) =  standardize(data.peaks{i}.intensity, s1, s2, pc1, pc2);
    end
    result.peaks = peaks;
end

function result = applyStandardizationToLandmarks(data, s1, s2)
    result = {};
    for i=1:size(data,2)
        result{i} = applyStandardizations(data{i}, s1, s2);
    end
end


function result = combineStandardizedLandmarks(data)
    minArr = data{1}.min;
    maxArr = data{1}.max;
    N = size(data,2);
    for i=2:N
        minArr = [minArr; data{i}.min];
        maxArr = [maxArr; data{i}.max];
    end
    result.min = minArr;
    result.max = maxArr;
    allPeaks = data{1}.peaks;
    for i=2:N
        allPeaks =[allPeaks; data{i}.peaks];
    end
    result.peaks = allPeaks;
end

function result = computeMeanLandmarks(data)
    result = struct();
    result.min=mean(data.min);
    result.max=mean(data.max);
    result.peaks = mean(data.peaks, 1);
end
%%
function imageData = loadImages(folder, trainingImageIndices)
    % Get all .nii files in the folder
    filePattern = fullfile(folder, '*.nii');
    niiFiles = dir(filePattern);
    
    % Initialize containers for info, images, and training labels
    infoData = {};
    images = {};
    isTraining = false(1, length(niiFiles));  % Preallocate logical array for training set
    imageIndex = zeros(1, length(niiFiles));  % Preallocate logical array for training set
    subjectIndex = zeros(1, length(niiFiles));  

    % Loop through all NIfTI files
    for i = 1:length(niiFiles)
        baseFileName = niiFiles(i).name;
        fullFileName = fullfile(folder, baseFileName);       
        % Extract the number after 'sub-' using regular expressions
        subject = regexp(fullFileName, '(?<=sub-)\d+', 'match');
        subject = int8(str2double(subject{1}));
        disp(['Training subject number: ', num2str(subject)]);

        % Store NIfTI info and read the image data
        infoData{i} = niftiinfo(fullFileName);    
        images{i} = niftiread(infoData{i});
        
        % Check if the image is part of the training set
        if ismember(subject, trainingImageIndices)
            disp(['Subject:' num2str(subject) ' in training set'])
            isTraining(i) = true;  % Mark as training image
        end
        subjectIndex(i) = subject;
        imageIndex(i) = i;
    end
    
    % Store the information in the output structure
    imageData.info = infoData';
    imageData.V = images';
    imageData.isTraining = isTraining';
    imageData.imageIndex = imageIndex';
    imageData.subjectIndex = subjectIndex';
end
%%
function data = setUpPeaks(data)
data.peaks{1} = [2 5];
data.peaks{2} = [2 20];
data.peaks{3} = [2 20];
data.peaks{4} = [2 20];
data.peaks{5} = [2 41];
data.peaks{6} = [2 6];
data.peaks{7} = [2 6];
data.peaks{8} = [2 10];
data.peaks{9} = [2 6];
data.peaks{10} = [2 6];
data.peaks{11} = [];
data.peaks{12} = [];
end

function data = setUpPercentiles(data)
for i = 1:length(data.subjectIndex) - 2
    data.percentiles{i} = [10 99.8];
end
data.percentiles{11} = [];
data.percentiles{12} = [];

end

%%
function landmarks = determineLandmarks(imageData, opt)

if ~isfield(opt, 'plotHistogram')
    opt.plotHistogram = false;
end
if ~isfield(opt, 'numBins')
    opt.numBins = 100;
end

landmarks = {};  
for i = 1:size(imageData.imageIndex, 1)
    subject = imageData.subjectIndex(i);
    subjectLabel = num2str(subject);
    if imageData.isTraining(i) == false
        disp(['Subject:' subjectLabel ' not in training set'])
        continue
    end
    disp(['Subject:' subjectLabel ' in training set'])

    % Access to the image
    Img = imageData.V{i};
      
    % Reshape the image matrix into a vector for percentile calculation
    flatImg = double(Img(:));  % Flatten the 3D image volume to 1D vector
    
    result = struct();
    result.min = min(flatImg);
    result.max = max(flatImg);
    histo = computeHistogram(flatImg, opt.numBins);
    result.peaks  = findNthLargestBinsIntensity(histo, imageData.peaks{i});
    result.percentiles = getPercentiles(flatImg,imageData.percentiles{i}); 
    result.imageIndex = imageData.imageIndex(i);
    result.subjectIndex = imageData.subjectIndex(i);
    landmarks{i} = result;

    
    % % Store the landmarks for this image
    % landmarks{i} = struct('index', imageIndices(i),...
    %     'min', minI, 'max', maxI, ...
    %     'pc1', pc1, 'pc2', pc2, 'mode', modeV);

    % Display percentiles for each image (optional)
    % subject = imageIndices(i);
    % disp(['Subject: ', num2str(subject)]);
    % disp(['Min:', num2str(minI),...
    %     '-max:', num2str(maxI), '-pc1:', num2str(pc1), ...
    %     '-pc2:', num2str(pc2), '-mode:', num2str(modeV)]);
    
    % % Find peaks and their prominence
    % options = struct();
    % options.prominenceThreshold = opt.prominenceThreshold(subject);
    % options.numBins = opt.numBins;
    % options.targetPeak = opt.targetPeak(subject);
    % options.binIdx = opt.binIdx(i);
    % peaks = findImagePeaks(subject, Img, options);
    % landmarks{i}.peaks = peaks;
    % if (opt.plotHistogram)
    %     % Plot the histogram and mark the landmarks
    %      histo = computeHistogram(Img, opt.numBins);
    %      plotHistogramWithLandmarks(subject, histo, landmarks{i})
    % end
end
end
%%
function result = getPercentiles(Img,percents)
    disp('Percentiles')
    disp(percents)
    percentiles = prctile(Img,percents);
    labels = [];
    values = [];
    for i=1:length(percentiles)
        labels{i} = num2str(percents(i));
        values(i) = percentiles(i);
    end
    result = struct();
    result.labels = labels;
    result.values = values;
end

%%
function result = computeHistogram(Img, numBins)
   [counts, edges] = histcounts(Img, numBins);  % Compute the histogram
   binCenters = (edges(1:end-1) + edges(2:end)) / 2;  % Midpoints of the bins
   result = struct();
   result.counts = counts;
   result.edges = edges;
   result.binCenters = binCenters;
end

%%
function results = findNthLargestBinsIntensity(histo, values)
    results = {};
    for i=1:length(values)
        results{i} = findNthLargestBinIntensity(histo, values(i));
    end
end

function result = findNthLargestBinIntensity(histo, n)
    
    % Sort the counts in descending order and get their indices
    [~, sortedIndices] = sort(histo.counts, 'descend');
    
    % Ensure n is within the valid range
    if n > length(sortedIndices)
        error('n is larger than the number of bins available.');
    end
    
    % Find the intensity value corresponding to the nth largest bin
    nthLargestBinIndex = sortedIndices(n);
    nthLargestBinIntensity = histo.binCenters(nthLargestBinIndex);
    
    % Get the count for this bin
    nthLargestBinCount = histo.counts(nthLargestBinIndex);
    
    % Store the result in a structure
    result = struct();
    result.intensity = nthLargestBinIntensity;  % Intensity value corresponding to nth largest bin
    result.count = nthLargestBinCount;          % Number of pixels in the nth largest bin
    result.bin = n;

    % Display the result
    disp(['Intensity value with the ', num2str(n), 'th largest bin: ', num2str(nthLargestBinIntensity)]);
    disp(['Number of pixels in this bin: ', num2str(nthLargestBinCount)]);
end

%%
function [s1, s2] = finds1s2(data)
s1 = 1;
s2 = 0;
for i=1:size(data,2)
    disp(['Max:' num2str(data{i}.max)])
    if data{i}.max > s2
        s2 = data{i}.max;
    end
end
end
%%

function logPeak(data, closestBin)
 disp(['Peak intensity:', num2str(data.value), ...
        '-Location:', num2str(data.loc), ...
        '-Prominence:', num2str(data.p),...
        '-Closest Bin:', num2str(closestBin)]);
end


function mappedImage = standardizeImage(Img, s1, s2, pc1, pc2)
    % Standardize the image using linear mapping between percentiles and [s1, s2]
    % Map the values between pC1 and pC2 to [s1, s2]
    
    % Perform linear mapping on the values within the range [pc1, pc2]
    ImgStandardized = s1 + (Img - pc1) .* (s2 - s1) ./ (pc2 - pc1);

    % Clip the values outside the range [pc1, pc2] to s1 and s2 respectively
    ImgStandardized(Img < pc1) = s1;
    ImgStandardized(Img > pc2) = s2;

    mappedImage = ImgStandardized;
end

function finalLandmarks = applyStandardization(imageSet, landmarks, s1, s2)
    % Apply standardization to all images in the set and compute final landmarks

    numImages = size(imageSet, 1);
    numLandmarks = length(landmarks{1}.peaks); % Number of landmarks/peaks

    % Array to store the standardized landmarks for each image
    standardizedLandmarks = zeros(numImages, numLandmarks);

    % Loop over all images
    for j = 1:numImages
        % Read the image and corresponding landmarks
        Img = imageSet{j};
        pc1 = landmarks{j}.pc1;
        pc2 = landmarks{j}.pc2;

        % Apply standardization to the image
        standardizedImage = standardizeImage(Img, s1, s2, pc1, pc2);

        % Compute new landmark values in the standardized image
        for k = 1:numLandmarks
            landmarkValue = landmarks{j}.peaks{k}.value;
            standardizedLandmarks(j, k) = s1 + (landmarkValue - pc1) .* (s2 - s1) ./ (pc2 - pc1);
        end
    end

    % Compute the final global mean landmarks
    finalLandmarks = mean(standardizedLandmarks, 1);
    disp('Final global landmarks:');
    disp(finalLandmarks);
end

function plotHistogramWithLandmarks(subject, histo, landmarks)
    % % Plot the histogram
    figure; % Create a new figure for each plot
    bar(histo.binCenters, histo.counts);  % Plot the histogram
    hold on;
    
    % Mark minI, maxI, pc1, pc2
    h1 = xline(landmarks.min, 'g', 'LineWidth', 2, 'Label', 'min');
    h2 = xline(landmarks.max, 'b', 'LineWidth', 2, 'Label', 'max');
    percentHandles = [];
    for i = 1:length(landmarks.percentiles.values)
        percentHandles(i) = xline(landmarks.percentiles.values(i), 'r', 'LineWidth', 2, ...
            'Label', ['Percentile (' num2str(landmarks.percentiles.labels{i}) ')']);
    end

    % Plot multiple peaks
    peakHandles = [];
    for i = 1:length(landmarks.peaks)
        peakHandles(i) = xline(landmarks.peaks{i}.intensity, 'm', 'LineWidth', 2, ...
            'Label', ['Peak (' num2str(landmarks.peaks{i}.bin) ')']);
    end

    % Add labels and title
    title(['Histogram with Landmarks, Subject: ', num2str(subject)]);
    xlabel('Intensity Value');
    ylabel('Count');

    % Create dynamic legend including peaks
    % legend([h1, h2, h3, h4, peakHandles], 'Histogram', 'min', 'max', ...
    %     'pc1', 'pc2', ...
    %     arrayfun(@(i) ['Peak ' num2str(i)], 1:length(landmarks.peaks), 'UniformOutput', false));

    hold off;
end


function plotHistogram(images, imageIndices)
figure;
t = tiledlayout(2, 3); % 2x3 grid

for i = 1:5
    image = images{i};  % Access the voxel data from the i-th image

    % Reshape the 3D image volume into a 1D vector for plotting the histogram
    image = image(:); 

    % Plot the histogram of the intensity values in the next tile
    nexttile;  % Move to the next tile in the layout
    histogram(image, 100);  % 100 bins for the histogram

    % Set title and labels for better understanding
    title(['Histogram of Subject ', num2str(imageIndices(i))]);
    xlabel('Intensity Value');
    ylabel('Count');
end

% Display the overall figure
grid on;
t.TileSpacing = 'compact';  % Adjust spacing between tiles if necessary
t.Padding = 'compact';  % Adjust padding for a cleaner look
end

function plotHistogram2(images)
figure;
t = tiledlayout(2, 3); % 2x3 grid

% Loop through the first 6 images to plot their histograms
for i = 1:6
    % Select the i-th image
    image = images{i};  % Access the voxel data from the i-th image

    % Use the first slice of the 3D image for plotting the histogram
    selectedImageSlice = image(:, :, 1);  % Taking the first slice (2D)
    normalizedImageSlice = mat2gray(selectedImageSlice);

    % Plot the histogram using imhist for the first slice
    nexttile;  % Move to the next tile in the layout
    imhist(normalizedImageSlice);  % imhist automatically uses 256 bins for grayscale images

    % Set title for better understanding
    title(['Histogram of Image ', num2str(i)]);
end

% Adjust layout spacing for better display
t.TileSpacing = 'compact';
t.Padding = 'compact';
end


function [pc0, pc998] = computePercentilesFromHistogram(image, numBins)
    % Compute the histogram
    [counts, edges] = histcounts(image(:), numBins);
    
    % Get bin centers (midpoints of the bins)
    binCenters = (edges(1:end-1) + edges(2:end)) / 2;

    % Compute cumulative histogram
    cumulativeCounts = cumsum(counts);
    
    % Normalize cumulative histogram to get the CDF (Cumulative Distribution Function)
    totalPixels = cumulativeCounts(end);  % Total number of pixels
    cumulativeCDF = cumulativeCounts / totalPixels;

    % Find the intensity values corresponding to 0th and 99.8th percentiles
    pc0 = binCenters(find(cumulativeCDF >= 0, 1, 'first'));    % 0th percentile
    pc998 = binCenters(find(cumulativeCDF >= 0.998, 1, 'first')); % 99.8th percentile

    % Display the results
    disp(['0th Percentile (Min): ', num2str(pc0)]);
    disp(['99.8th Percentile: ', num2str(pc998)]);
end

%%
function displayLandmarks(landmarks)
    % Function to display the contents of the 'landmarks' cell array
    
    % Loop through each landmark
    for i = 1:length(landmarks)
        fprintf('\nLandmark %d:\n', i);
        
        % Access the current landmark struct
        landmark = landmarks{i};
        
        % Display 'min' and 'max' fields
        fprintf('  Min intensity: %g\n', landmark.min);
        fprintf('  Max intensity: %g\n', landmark.max);
        
        % Display 'imageIndex' and 'subjectIndex'
        fprintf('  Image Index: %d\n', landmark.imageIndex);
        fprintf('  Subject Index: %d\n', landmark.subjectIndex);
        
        % Display percentiles
        fprintf('  Percentiles:\n');
        disp(landmark.percentiles)
        % for j=1:length(landmark.percentiles.values)
        %     fprintf('    %s-%g\n', landmark.percentiles.labels{j},...
        %          landmark.percentiles.values(j));
        % end
        
        % Display peaks
        fprintf('  Peaks:\n');
        %disp(landmark.peaks)
        for j = 1:length(landmark.peaks)
             peak = landmark.peaks{j};
             fprintf('    Peak %d:\n', j);
             fprintf('      Intensity: %g\n', peak.intensity);
        %     fprintf('      Count: %d\n', peak.count);
        %     fprintf('      Bin: %d\n', peak.bin);
        end
    end
end
%%
function displayStandardizedLandmarks(landmarks)
    % Function to display the contents of the 'standardizedLandmarks' cell array
    
    % Loop through each standardized landmark
    for i = 1:length(landmarks)
        fprintf('\nStandardized Landmark %d:\n', i);
        
        % Access the current standardized landmark struct
        landmark = landmarks{i};
        
        % Display 'min' and 'max' fields
        fprintf('  Min intensity: %g\n', landmark.min);
        fprintf('  Max intensity: %g\n', landmark.max);
        
        % Display peaks
        fprintf('  Peaks:\n');
        % Loop through each element in 'peaks'
        for j = 1:length(landmark.peaks)
            fprintf('    Peak %d: %g\n', j, landmark.peaks(j));
        end
    end
end
%%
function displayCombinedAggregatedLandmarks(data)
    % Function to display the contents of the aggregated staandardized 'landmarks'
    disp('Min Intensity')
    disp(data.min)
    disp('Max Intensity')
    disp(data.max)
    disp('Peaks')
    disp(data.peaks)
end
%
