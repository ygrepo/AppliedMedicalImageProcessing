% Add NIfTI toolbox to the MATLAB path
clearvars
%%
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
landmarks.percentiles = getPercentiles(flatImg,[10 99.8]); 
subject = imageData.subjectIndex(subjectIndex);
plotHistogramWithLandmarks(subject, histo, landmarks);

%% Set up the landmark parameters
imageData = setUpPeaks(imageData);
imageData = setUpPercentiles(imageData);

%%
clc
opt = struct();
opt.plotHistogram = false;
opt.numBins = 100;
imageLandmarks = determineLandmarks(imageData, opt);
%%
clc;
displayLandmarks(imageLandmarks)
%%
clc
[s1, s2] = finds1s2(imageLandmarks);
% adjust s2 to be less than max. of intensities
%
%%
%applyStandardizations(imageLandmarks{1}, s1, s2)
imageLandmarks = applyStandardizationToLandmarks(imageLandmarks, s1, s2);
%%
clc
displayLandmarks(imageLandmarks);
%%
clc
% combineStandardizedLandmarks(imageLandmarks)
% computeMeanLandmarks(imageLandmarks)
imageLandmarks = addMeanStandardScaledLandmarks(imageLandmarks);
%%
clc
displayLandmarks(imageLandmarks);
%%
%imageLandmarks{1} = transformImage(imageLandmarks{1},s1, s2);
%%
imageLandmarks = transformImages(imageLandmarks, s1, s2, opt);
%%
opt = struct();
opt.numBins = 20;
opt.xlim = [s1 s2];
plotBothHistograms(imageLandmarks, s1, s2, opt);
%%
%
figure;
sliceViewer(imageLandmarks{1}.V, []);
title('Subject 2: Original Image');
%%
clc
% Define the grayscale window (e.g., [low, high] intensity values)
window = [50, 200];  % Desired intensity window
opt = struct();
opt.numCols =5;
subjectImg = imageLandmarks{12};
titleText = ['Subject ' num2str(subjectImg.subjectIndex)];
plotSlices(subjectImg.V, window, titleText, opt);

%%
% Define the grayscale window (e.g., [low, high] intensity values)
window = [20, 100];  % Desired intensity window
opt = struct();
opt.numCols =5;
for i=1:size(imageLandmarks, 2)
    indexSubject = 5;
    subjectImg = imageLandmarks{i};
    titleText = ['Subject ' num2str(subjectImg.subjectIndex)];
    plotSlices(subjectImg.V, window, titleText, opt)
end
%%
opt = struct();
opt.window = [20, 100];
opt.slice = 120;
opt.original = true;
plotSliceCrossSubjects(imageLandmarks, opt)

%%
opt = struct();
opt.window = [20, 100];
opt.slice = 120;
opt.original = false;
plotSliceCrossSubjects(imageLandmarks, opt)

%%
%
figure;
sliceViewer(imageLandmarks{2}.V, []);
title('Subject 3: Original Image');
% 
% figure;
% sliceViewer(imageLandmarks{1}.transfV, []);
% title('Transformed Image');

%%
function res = standardize(val, s1, s2, pc1, pc2)
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
    pc1 = data.pc1;
    pc2 = data.pc2;
    values = zeros(length(data.landmarks),1);
    for i=1:length(data.landmarks)
        values(i) =  standardize(data.landmarks(i), s1, s2, pc1, pc2);
    end
    data.standardScaledlandmarks = values;
end

function data = applyStandardizationToLandmarks(data, s1, s2)
    for i=1:size(data,2)
        data{i} = applyStandardizations(data{i}, s1, s2);
    end
end


function arr = combineStandardizedLandmarks(data)
    arr = data{1}.standardScaledlandmarks';
    N = size(data,2);
    for i=2:N
        if data{i}.isTraining
            disp(['Adding landmarks of ',  num2str(data{i}.subjectIndex) ' as part of training'])
            arr = [arr; data{i}.standardScaledlandmarks'];
        else
            disp(['Skipping landmarks of ',  num2str(data{i}.subjectIndex) ' as not part of training'])
        end
    end
end

function arr = computeMeanLandmarks(data)
    arr = combineStandardizedLandmarks(data);
    arr = mean(arr, 1);
end

function data = addMeanStandardScaledLandmarks(data)
    meanScaledValues = computeMeanLandmarks(data);
    for i=1:size(data,2)
        data{i}.meanStandardScaledlandmarks = meanScaledValues;
    end
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
data.peaks{11} = [2 6];
data.peaks{12} = [2 6];
end

function data = setUpPercentiles(data)
for i = 1:length(data.subjectIndex)
    data.percentiles{i} = [10 99.8];
end
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
    disp(['Subject:' subjectLabel]);
    % Access to the image
    Img = imageData.V{i};
      
    % Reshape the image matrix into a vector for percentile calculation
    flatImg = double(Img(:));  % Flatten the 3D image volume to 1D vector
    
    result = struct();
    values = [];
    result.min = min(flatImg);
    if result.min < 0 
        error('Min is negative.');
    end
    values(1) =  result.min;
    histo = computeHistogram(flatImg, opt.numBins);
    peaks  = findNthLargestBinsIntensity(histo, imageData.peaks{i});
    values = [values; peaks];
    percentiles = getPercentiles(flatImg,imageData.percentiles{i}); 
    result.pc1 = percentiles.values(1);
    if result.pc1 < 0 
        error('PC1 is negative.');
    end
    result.pc2 = percentiles.values(end);
    if result.pc2 < 0 
        error('PC2 is negative.');
    end
    values = [values; percentiles.values];
    result.max = max(flatImg);
    if result.max < 0 
        error('Max is negative.');
    end
    values = [values; result.max];
    result.landmarks = sort(values);
    result.imageIndex = imageData.imageIndex(i);
    result.subjectIndex = imageData.subjectIndex(i);
    result.V = imageData.V{i};
    result.isTraining = imageData.isTraining(i);
    result.histo = histo;
    landmarks{i} = result;
end
end
%%
function result = getPercentiles(Img,percents)
    disp('Percentiles')
    disp(percents)
    percentiles = prctile(Img,percents);
    labels = [];
    values = zeros(length(percentiles),1);
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

function result = findNthLargestBinsIntensity(histo, values)
    result = zeros(length(values),1);
    for i=1:length(values)
         peak = findNthLargestBinIntensity(histo, values(i));
         result(i) = peak.intensity;
    end
end

%%
function data = transformImage(data, s1, s2, opt)
   % Access to the image
   Img = data.V;
   
   % Get the original size of the image
   originalSize = size(Img);
      
   % Reshape the image matrix into a vector for percentile calculation
   flatImg = double(Img(:));  % Flatten the 3D image volume to 1D vector

    % Initialize transformed image Vsi with the same size as Vi
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
   mask = (flatImg < data.pc1);
   transfV(mask) = s1;
   mask = (flatImg > data.pc2);
   transfV(mask) = s2;
   
   % Reshape transfV back into the original image dimensions
   data.transfV = reshape(transfV, originalSize);
   data.transfVHisto = computeHistogram(double(transfV(:)), opt.numBins);
end

function data = transformImages(data, s1, s2, opt)
 for i=1:size(data, 2)
     data{i} = transformImage(data{i}, s1, s2, opt);
 end
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
if s2 < s1
    error('S2 < S1');
end
end
%%
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
        peakHandles(i) = xline(landmarks.peaks(i), 'm', 'LineWidth', 2, ...
            'Label', ['Peak (' num2str(landmarks.peakIndex(i)) ')']);
    end

    % Add labels and title
    title(['Histogram with Landmarks, Subject: ', num2str(subject)]);
    xlabel('Intensity Value');
    ylabel('Count');
    hold off;
end

function plotBothHistograms(data, s1, s2, opt)
    N = size(data, 2);
    
    % First set of histograms
    figure; % Create a new figure for original histograms
    for i = 1:N
        histogram(data{i}.V, 'BinLimits', [s1, s2], 'NumBins', opt.numBins);  % Plot the histogram
        hold on;
    end
    title('Original Histograms', 'FontSize', 24,'FontWeight','bold');
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
    title('Transformed Histograms', 'FontSize', 24,'FontWeight','bold');
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
window = opt.window;  
slice  = opt.slice;
figure;
t = tiledlayout(3, 4);
titleText = ['Slice ' num2str(slice)];
title(t, titleText);

%title(t, titleText);
for i=1:size(data, 2)
    imgData = data{i};
    nexttile; 
    if opt.original
        imshow(imgData.V(:,:,slice), [], 'DisplayRange', window);
    else
        imshow(imgData.transfV(:,:,slice), [], 'DisplayRange', window);
    end
    titleText = ['Subject ' num2str(imgData.subjectIndex)];
    title(titleText);
end

end

%%
function displayLandmarks(imgLandmarks)
    % Function to display the contents of the 'landmarks' cell array
    
    % Loop through each landmark
    for i = 1:size(imgLandmarks, 2)        
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

