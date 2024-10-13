% Add this file to the MATLAB path with the 
clearvars
%% Download the images from the folder containing the NIfTI images ----
clc
% Specify the directory containing the .nii.gz files
folder = './MRI_T1W';  % Update this path to your directory
trainingImageIndices = [2:7, 9:12];
imageData = loadImages(folder, trainingImageIndices);
%% Tune the bin number for peak landmarks ----
subjectIndex = 12;
img = imageData.V{subjectIndex};
landmarks = struct();
flatImg = img(:);
histo = computeHistogram(flatImg, 100);
landmarks.min = min(flatImg);
landmarks.max = max(flatImg); 
landmarks.peakIndex = [2 6];
landmarks.peaks  = findNthLargestBinsIntensity(histo, landmarks.peakIndex);
landmarks.percentiles = computePercentiles(flatImg,[10 20 30 60 70 99.8]); 
subject = imageData.subjectIndex(subjectIndex);
plotHistogramWithLandmarks(subject, histo, landmarks);

%% Set up the landmark parameters ----
imageData = setUpPeaks(imageData);
imageData = setUpPercentiles(imageData);

%% Compute landmarks ----
clc
opt = struct();
opt.numBins = 100;
imageLandmarks = computeLandmarks(imageData, opt);
%%
clc;
displayLandmarks(imageLandmarks)
%%
clc
[s1, s2] = finds1s2(imageLandmarks);
%
%% Standardization Step ----
%applyStandardizations(imageLandmarks{1}, s1, s2)
imageLandmarks = applyStandardizationToLandmarks(imageLandmarks, s1, s2);
%%
clc
displayLandmarks(imageLandmarks);
%% Standardization Step ----
clc
% combineStandardizedLandmarks(imageLandmarks)
% computeMeanLandmarks(imageLandmarks)
imageLandmarks = addMeanStandardScaledLandmarks(imageLandmarks);
%%
clc
displayLandmarks(imageLandmarks);
%%
%imageLandmarks{1} = transformImage(imageLandmarks{1},s1, s2);
%% Transformation/Standard Scaling Step ----
opt = struct();
opt.numBins = 100;
imageLandmarks = transformImages(imageLandmarks, s1, s2, opt);
%% Evaluation Step ----
opt = struct();
opt.numBins = 20;
opt.xlim = [s1 s2];
plotBothHistograms(imageLandmarks, s1, s2, opt);
%% Plot a specific slice across all subjects before transformation ----
opt = struct();
opt.window = [20, 100];
opt.slices = [110, 110, 116, 101, 99, 116, 112, 263, 81, 149, 158, 136];
opt.original = true;
plotSliceCrossSubjects(imageLandmarks, opt)

%% Plot a specific slice across all subjects after transformation ----
% We want to be the same slice as the previous section.
opt = struct();
opt.window = [20, 200];
opt.slices = [110, 110, 116, 101, 99, 116, 112, 263, 81, 149, 158, 136];
opt.original = false;
plotSliceCrossSubjects(imageLandmarks, opt)

%% 
% 
% opt.original = true;
% plotSlice(imageLandmarks{12}, opt)

%% Function to Download the images ----
function imageData = loadImages(folder, trainingImageIndices)
    % Get all .nii files in the folder
    filePattern = fullfile(folder, '*.nii');
    niiFiles = dir(filePattern);
    
    % Initialize containers for info, images, and training labels
    infoData = {};
    images = {};
    isTraining = false(1, length(niiFiles));  % Preallocate logical array for training set
    imageIndex = zeros(1, length(niiFiles));  % Preallocate index array for image indexing
    subjectIndex = zeros(1, length(niiFiles));  % Preallocate logical array for image subject references

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
%% Setup peaks and percentile functions ----
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
data.peaks{11} = [2 6];
data.peaks{12} = [2 6];
end

function data = setUpPercentiles(data)
for i = 1:length(data.subjectIndex)
    data.percentiles{i} = [10 20 30 60 70 99.8];
end

end

%% Function to determine for all the images their landmarks ----
function landmarks = computeLandmarks(imageData, opt)

    if ~isfield(opt, 'numBins')
        opt.numBins = 100;
    end

    numImages = size(imageData.imageIndex, 1);  % Number of images
    numPeaks = length(imageData.peaks{1});  % Assuming all images have the same number of peaks
    numPercentiles = length(imageData.percentiles{1});  % Assuming all images have the same number of percentiles

    landmarks = cell(numImages, 1);  

    for i = 1:numImages
        subject = imageData.subjectIndex(i);
        subjectLabel = num2str(subject);
        disp(['Subject: ' subjectLabel]);

        % Access the image and flatten it for processing
        Img = double(imageData.V{i});
        flatImg = Img(:);  

        % Initialize the result structure
        result = struct();

        % Preallocate the 'values' array based on the number of landmarks
        values = zeros(1, 2 + numPeaks + numPercentiles);  % min + max + peaks + percentiles

        % Compute min and check
        result.min = min(flatImg);
        if result.min < 0 
            error('Min is negative.');
        end
        values(1) = result.min;

        % Compute the histogram and peaks
        histo = computeHistogram(flatImg, opt.numBins);
        peaks = findNthLargestBinsIntensity(histo, imageData.peaks{i});
        values(2:(1 + numPeaks)) = peaks;  % Assign peaks

        % Compute percentiles
        percentiles = computePercentiles(flatImg, imageData.percentiles{i});
        result.pc1 = percentiles.values(1);
        result.pc2 = percentiles.values(end);

        % Error checks for percentiles
        if result.pc1 < 0 
            error('PC1 is negative.');
        end
        if result.pc2 < 0 
            error('PC2 is negative.');
        end
        values((2 + numPeaks):(1 + numPeaks + numPercentiles)) = percentiles.values;  % Assign percentiles

        % Compute max and check
        result.max = max(flatImg);
        if result.max < 0 
            error('Max is negative.');
        end
        values(end) = result.max;

        % Sort landmarks
        result.landmarks = sort(values);

        % Assign other metadata fields
        result.imageIndex = imageData.imageIndex(i);
        result.subjectIndex = imageData.subjectIndex(i);
        result.V = imageData.V{i};
        result.isTraining = imageData.isTraining(i);
        result.histo = histo;

        % Store the result in the landmarks array
        landmarks{i} = result;
    end
end

%% Function to compute the percentiles given a set of percents labels ----
function result = computePercentiles(Img, percents)
    % Display the percentiles
    disp('Percentiles:')
    disp(percents)
    
    % Compute the percentiles
    percentiles = prctile(Img, percents);
    
    % Preallocate the result structure and initialize labels and values
    numPercentiles = length(percentiles);
    labels = cell(numPercentiles, 1); 
    values = zeros(numPercentiles, 1); 
    
    % Loop through each percentile and store the values and labels
    for i = 1:numPercentiles
        labels{i} = num2str(percents(i));  % Convert each percent to a string
        values(i) = percentiles(i);        % Store the corresponding percentile value
    end
    
    % Create the result structure
    result = struct();
    result.labels = labels;
    result.values = values;
end


%% Function to compute an image histogram ----
function result = computeHistogram(Img, numBins)
   [counts, edges] = histcounts(Img, numBins);  % Compute the histogram
   binCenters = (edges(1:end-1) + edges(2:end)) / 2;  % Midpoints of the bins
   result = struct();
   result.counts = counts;
   result.edges = edges;
   result.binCenters = binCenters;
end

%% Functions to find the peaks of an histogram ----
function result = findNthLargestBinIntensity(histo, n)
    % Function to display the contents of the 'landmarks' cell array
    % histo: histogram structure containing the bin center locations and
    % the counts in each bins.
    % N: Bin index of the sorted bins by counts to look for.
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

function result = findNthLargestBinsIntensity(histo, values)
% helper functions to compute the peaks for a given image histogram.
    result = zeros(length(values),1);
    for i=1:length(values)
         peak = findNthLargestBinIntensity(histo, values(i));
         result(i) = peak.intensity;
    end
end

%% Functions to transform the image histogram using the standard scale ----
function data = transformImage(data, s1, s2, opt)
   % Access to the image
   Img = data.V;
   
   % Get the original size of the image
   originalSize = size(Img);
      
   % Reshape the image matrix into a vector for percentile calculation
   flatImg = double(Img(:));  

    % Initialize transformed image transfV with the same size as V
   transfV = zeros(size(flatImg));
   for i=1: length(data.landmarks)-1
        % Get intensity range [mu_i(k), mu_i(k+1)]
        % Find voxels within the range for this section of the scale
        mu1 = data.landmarks(i);
        mu2 = data.landmarks(i+1);
        mu1mean = data.meanStandardScaledlandmarks(i);
        mu2mean = data.meanStandardScaledlandmarks(i+1);
        mask = (mu1 <= flatImg) & (flatImg < mu2);
        transfV(mask) = (flatImg(mask) - mu1) * (mu2mean - mu1mean)/ (mu2-mu1);
   end
   % Handle pixels less than pc1 and pc2.
   mask = (flatImg < data.pc1);
   transfV(mask) = s1;
   mask = (flatImg > data.pc2);
   transfV(mask) = s2;
   
   % Reshape transfV back into the original image dimensions
   data.transfV = reshape(transfV, originalSize);
   data.transfVHisto = computeHistogram(double(transfV(:)), opt.numBins);
end

function data = transformImages(data, s1, s2, opt)
 for i=1:size(data, 1)
     fprintf('Subject:%g\n', data{i}.subjectIndex);
     data{i} = transformImage(data{i}, s1, s2, opt);
 end
end

%% Function to find S1 and S2 ----
% S1 is set to 1
% S2 is the max of all the intensities across all the images.
function [s1, s2] = finds1s2(data)
s1 = 1;
s2 = 0;
for i=1:size(data,1)
    disp(['Max:' num2str(data{i}.max)])
    if data{i}.max > s2
        s2 = data{i}.max;
    end
end
if s2 < s1
    error('S2 < S1');
end
end
%% Standardization functions ----

function res = standardize(val, s1, s2, pc1, pc2)
% Function to standardize one value based on s1, s2, pc1 and pc2.
    if val < pc1
        fprintf('Val:%g less than pc1:%g\n', val, pc1);
        res = s1;
        fprintf('Val:%g\n', val);
        return
    end
    if val > pc2
        fprintf('Val:%g greater than pc2:%g\n', val, pc2);
        res = s2;
        fprintf('Val:%g\n', res);
        return
    end
    res = s1 + (val - pc1) * (s2 - s1) / (pc2 - pc1);
end

function data = applyStandardizations(data, s1, s2)
% Function to standardize one image landmarks.
    pc1 = data.pc1;
    pc2 = data.pc2;
    values = zeros(length(data.landmarks),1);
    for i=1:length(data.landmarks)
        values(i) =  standardize(data.landmarks(i), s1, s2, pc1, pc2);
    end
    data.standardScaledlandmarks = values;
end

function data = applyStandardizationToLandmarks(data, s1, s2)
% Function to standardize the landmarks of all images.
    for i=1:size(data,1)
        fprintf('Subject:%g\n', data{i}.subjectIndex);
        data{i} = applyStandardizations(data{i}, s1, s2);
    end
end


function arr = combineStandardizedLandmarks(data)
    % Function to aggregate all the standardized landmarks into a matrix
    % Rows correspond to the landmarks of each image.
    % Skips images that are not part of the training dataset.

    % Get the number of data entries and the size of the landmarks
    N = size(data, 1);
    numLandmarks = length(data{1}.standardScaledlandmarks);

    % Preallocate arr based on the number of training images
    arr = zeros(N, numLandmarks);  % Preallocate for the worst case (all are training)
    
    % Initialize row counter for arr
    rowCount = 0;

    % Loop through the data
    for i = 1:N
        if data{i}.isTraining
            disp(['Adding landmarks of ', num2str(data{i}.subjectIndex), ' as part of training'])
            
            % Increment the row counter
            rowCount = rowCount + 1;
            
            % Assign the standardized landmarks to the next row
            arr(rowCount, :) = data{i}.standardScaledlandmarks';
        else
            disp(['Skipping landmarks of ', num2str(data{i}.subjectIndex), ' as not part of training'])
        end
    end

    % Trim the preallocated array to the actual number of training images
    arr = arr(1:rowCount, :);  % Keep only the filled rows
end

function arr = computeMeanLandmarks(data)
% Function which combines all the standardized image landmarks into
% a matrix and then compute their mean column-wise.
    arr = combineStandardizedLandmarks(data);
    arr = mean(arr, 1);
end

function data = addMeanStandardScaledLandmarks(data)
% Function add to the image data structure the mean of all the standarized
% landmarks.
    meanScaledValues = computeMeanLandmarks(data);
    for i=1:size(data,1)
        data{i}.meanStandardScaledlandmarks = meanScaledValues;
    end
end

%% Plotting Histogram Functions ----
function plotHistogramWithLandmarks(subject, histo, landmarks)
    % Plot the histogram of a subject with landmarks
    figure; 
    bar(histo.binCenters, histo.counts);  % Plot the histogram
    hold on;

    % Use colormap to assign colors dynamically
    cmap = lines(length(landmarks.peaks) + length(landmarks.percentiles.values) + 2);  

    % Mark min, max, pc1, pc2 using colormap colors
    h1 = xline(landmarks.min, 'Color', cmap(1,:), 'LineWidth', 2, 'Label', 'min');
    h2 = xline(landmarks.max, 'Color', cmap(2,:), 'LineWidth', 2, 'Label', 'max');

    % Initialize an array to store handles for the legend
    legendHandles = [h1, h2];
    legendLabels = {'min', 'max'};
    
    % Plot percentile lines
    for i = 1:length(landmarks.percentiles.values)
        h = xline(landmarks.percentiles.values(i), 'Color', cmap(i+2,:), 'LineWidth', 2, ...
            'Label', ['Percentile (' num2str(landmarks.percentiles.labels{i}) ')']);
        legendHandles(end+1) = h;
        legendLabels{end+1} = ['Percentile (' num2str(landmarks.percentiles.labels{i}) ')'];
    end

    % Plot peak lines
    for i = 1:length(landmarks.peaks)
        h = xline(landmarks.peaks(i), 'Color', cmap(i+2+length(landmarks.percentiles.values),:), 'LineWidth', 2, ...
            'Label', ['Peak (' num2str(landmarks.peakIndex(i)) ')']);
        legendHandles(end+1) = h;
        legendLabels{end+1} = ['Peak (' num2str(landmarks.peakIndex(i)) ')'];
    end

    % Add labels and title
    title(['Subject: ', num2str(subject) ':Histogram with Landmarks'],...
        'FontSize', 24, 'FontWeight', 'bold');
    xlabel('Intensity Value', 'FontSize', 24, 'FontWeight', 'bold');
    ylabel('Count', 'FontSize', 24, 'FontWeight', 'bold');
    
    % Add the legend
    legend(legendHandles, legendLabels, 'Location', 'best', 'FontSize', 14);
    set(gca, 'FontSize', 14); 

    hold off;
end


function plotBothHistograms(data, s1, s2, opt)
    N = size(data, 1);
    
    % First set of histograms
    figure; % Create a new figure for original histograms
    for i = 1:N
        histogram(data{i}.V, 'BinLimits', [s1, s2], 'NumBins', opt.numBins);  % Plot the histogram
        hold on;
    end
    title('Original Image Histograms', 'FontSize', 24,'FontWeight','bold');
    xlabel('Intensity','FontSize', 24,'FontWeight','bold');
    ylabel('Count', 'FontSize', 24,'FontWeight','bold');
    
    % Adjust the axis limits for zooming (optional)
    if isfield(opt, 'xlim')
        xlim(opt.xlim);
    end
    if isfield(opt, 'ylim')
        ylim(opt.ylim);
    end
    set(gca, 'FontSize', 14); 

    % Second set of histograms (for transformed image)
    figure; % Create a new figure for transformed histograms
    for i = 1:N
        histogram(data{i}.transfV, 'BinLimits', [s1, s2], 'NumBins', opt.numBins);  % Plot the histogram 
        hold on;
    end
    title('Transformed Image Histograms', 'FontSize', 24,'FontWeight','bold');
    xlabel('Intensity','FontSize', 24,'FontWeight','bold');
    ylabel('Count', 'FontSize', 24,'FontWeight','bold');
    
    % Adjust the axis limits for zooming (optional)
    if isfield(opt, 'xlim')
        xlim(opt.xlim);
    end
    if isfield(opt, 'ylim')
        ylim(opt.ylim);
    end
    set(gca, 'FontSize', 14); 

end
%% Image Slice Plotting Functions ----

function plotSlice(data, opt)
if ~isfield(opt, 'original')
    opt.original = true;
end

figure;
if opt.original
    sliceViewer(data.V, []);
    titleText = ['Subject ' num2str(data.subjectIndex) ' Before Transformation'];
else
     sliceViewer(data.transfV, []);
    titleText = ['Subject ' num2str(data.subjectIndex) ' Before Transformation'];
end
title(titleText, 'FontSize', 16,'FontWeight','bold');
end

function plotSlices(V, window, titleText, opt)
if ~isfield(opt, 'numCols')
    opt.numCols = 3;
end
totalSlices = 2 * opt.numCols;

% Create a tiled layout with 2 rows and numCol columns
figure;
t = tiledlayout(2, opt.numCols);
title(t, titleText);

% Loop through a few slices and display them with a specific grayscale window
for i = 1:totalSlices  % Display slices
    sliceIndex = i * 20;  % Choose every 20th slice
    nexttile;  
    imshow(V(:,:,sliceIndex), [], 'DisplayRange', window);
    title(['Slice ', num2str(sliceIndex)]);
end

end


function plotSliceCrossSubjects(data, opt)
if ~isfield(opt, 'original')
    opt.original = true;
end

window = opt.window;  
slices  = opt.slices;
figure;
t = tiledlayout(3, 4);
if opt.original
    titleText = 'Before Transformation';
else
    titleText = 'After Transformation';
end

title(t, titleText, 'FontSize', 24,'FontWeight','bold');

for i=1:size(data, 1)
    imgData = data{i};
    nexttile; 
    slice = slices(i);
    if opt.original
        imshow(imgData.V(:,:,slice), [], 'DisplayRange', window);
    else
        imshow(imgData.transfV(:,:,slice), [], 'DisplayRange', window);
    end
    titleText = ['Subject:' num2str(imgData.subjectIndex) '-Slice' num2str(slice)];
    title(titleText, 'FontSize', 24,'FontWeight','bold');
end

end

%% Logging function of Image landmarks and other metadata information ----
function displayLandmarks(imgLandmarks)
    % Function to display the contents of the 'landmarks' cell array
    
    % Loop through each landmark
    for i = 1:size(imgLandmarks, 1)        
        % Access the current landmark struct
        landmark = imgLandmarks{i};
        %disp(landmark.subjectIndex)
        fprintf('Subject:%g\n', landmark.subjectIndex);

        % Display isTraining fields
        fprintf('  Training: %g\n', landmark.isTraining);

        % Display 'min' and 'max' fields
        fprintf('  Min intensity: %g\n', landmark.min);
        fprintf('  Max intensity: %g\n', landmark.max);

        % Display 'pc1' and 'pc2' fields
        fprintf('  pc1: %g\n', landmark.pc1);
        fprintf('  pc2: %g\n', landmark.pc2);

        % Display landmarks
        fprintf('  Landmarks:\n');
        disp(landmark.landmarks)
        
        if isfield(landmark, 'standardScaledlandmarks')
            fprintf('  Standard Scaled Landmarks:\n');
            disp(landmark.standardScaledlandmarks)
        end
        if isfield(landmark, 'meanStandardScaledlandmarks')
            fprintf('  Mean Standard Scaled Landmarks:\n');
            disp(landmark.meanStandardScaledlandmarks)
        end

    end
end

