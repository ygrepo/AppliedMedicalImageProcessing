% Add NIfTI toolbox to the MATLAB path
clearvars
%%
clc
% Specify the directory containing the .nii.gz files
folder = './MRI_T1W';  % Update this path to your directory
trainingImageIndices = [2:7, 9:12];
imageData = loadImages(folder, trainingImageIndices);
%imageData.V(imageData.isTraining)

%%

images = imageData.V(imageData.isTraining);
setIndex = imageData.imageIndex(imageData.isTraining);
plotHistogram(images(1:5),setIndex(1:5));
plotHistogram(images(6:end),setIndex(6:end));

%%

plotHistogram2(imageData.V(imageData.isTraining))
%%
imageData.info{1}
%%
opt = struct();
opt.plotHistogram = true;
opt.prominenceThreshold = [10, 10, 10];
opt.targetPeak = [100, 200, 10];
landmarks = determineLandmarks(imageData.V(imageData.isTraining),...
    imageData.imageIndex(imageData.isTraining), opt);


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
    
    % Loop through all NIfTI files
    for i = 1:length(niiFiles)
        baseFileName = niiFiles(i).name;
        fullFileName = fullfile(folder, baseFileName);       
        % Extract the number after 'sub-' using regular expressions
        subjectNumber = regexp(fullFileName, '(?<=sub-)\d+', 'match');
        % Convert to numeric 
        subjectNumber = str2double(subjectNumber{1});
        disp(['Training subject number: ', num2str(subjectNumber)]);

        % Store NIfTI info and read the image data
        infoData{i} = niftiinfo(fullFileName);    
        images{i} = niftiread(infoData{i});
        
        % Check if the image is part of the training set
        if ismember(subjectNumber, trainingImageIndices)
            isTraining(subjectNumber) = true;  % Mark as training image
        end
        imageIndex(i) = i;
    end
    
    % Store the information in the output structure
    imageData.info = infoData';
    imageData.V = images';
    imageData.isTraining = isTraining';
    imageData.imageIndex = imageIndex';
end

function landmarks = determineLandmarks(imageSet,...
    imageIndices, opt)

if ~isfield(opt, 'prominenceThreshold')
    opt.prominenceThreshold = 10;
end
if ~isfield(opt, 'plotHistogram')
    opt.plotHistogram = false;
end
if ~isfield(opt, 'numBins')
    opt.numBins = 100;
end


landmarks = {};  
for i = 2:2
%for i = 1:size(imageSet, 1)

    % Read the NIfTI data
    Img = imageSet{i};
      
    % Reshape the image matrix into a vector for percentile calculation
    Img = double(Img(:));  % Flatten the 3D image volume to 1D vector
    
    minI = min(Img);
    maxI = max(Img);
    
    % Calculate pc1 and pc2
    pc1 = prctile(Img, 10);
    pc2 = prctile(Img, 99.8); 
    
    % Mode
    modeV = mode(Img);
    
    % Store the landmarks for this image
    landmarks{i} = struct('index', imageIndices(i),...
        'min', minI, 'max', maxI, ...
        'pc1', pc1, 'pc2', pc2, 'mode', modeV);

    % Display percentiles for each image (optional)
    disp(['Image: ', num2str(i), '-minI:', num2str(minI),...
        '-maxI:', num2str(maxI), '-pc1:', num2str(pc1), ...
        '-pc2: ', num2str(pc2), '-mode:', num2str(modeV)]);
    
    % Find peaks and their prominence
    %[pks, locs, ~, prominence] = findpeaks(V);
    subject = imageIndices(i);
    options = struct();
    options.prominenceThreshold = opt.prominenceThreshold(subject);
    options.numBins = opt.numBins;
    options.targetPeak = opt.targetPeak(subject);
    peaks = findImagePeaks(subject, Img, options);
    landmarks{i}.peaks = peaks;
    if (opt.plotHistogram)
        % Plot the histogram and mark the landmarks
         histo = computeHistogram(Img, opt.numBins);
         plotHistogramWithLandmarks(subject, histo, landmarks{i})
    end
end
end

function peaks = findImagePeaks(subject, Img,opt)

    % Find peaks and their prominences
    [pks, locs, ~, prominence] = findpeaks(Img);

    % Filter peaks based on prominence greater than the threshold
    prominenceThreshold = opt.prominenceThreshold;
    validPeaks = pks(prominence >= prominenceThreshold);
    validLocs = locs(prominence >= prominenceThreshold);
    validProminences = prominence(prominence >= prominenceThreshold);
    
    peaks = {};
    % Check if there are at least two valid peaks
    if length(validPeaks) >= 2
         result = findpeakByGivenIndex(Img, validLocs, validProminences, 2);
        disp(['Subject: ', num2str(subject)]);
        histo = computeHistogram(Img, opt.numBins);
        closestBin = getClosestBin(result, histo);
        logPeak(result, closestBin);
        peaks{1} = result;

        result = findPeakCloserToIntensity(Img,...
            validLocs, validProminences, opt.targetPeak);
        closestBin = getClosestBin(result, histo);
        logPeak(result, closestBin);
        peaks{2} = result;     
    else
        disp('There are less than two valid peaks with the specified prominence.');
    end
end

function result = computeHistogram(Img, numBins)
   [counts, edges] = histcounts(Img, numBins);  % Compute the histogram
   binCenters = (edges(1:end-1) + edges(2:end)) / 2;  % Midpoints of the bins
   result = struct();
   result.counts = counts;
   result.edges = edges;
   result.binCenters = binCenters;
end

function result = findpeakByGivenIndex(Im, locs, prominences, idx)
% Sort the valid peaks by their intensity values 
[sortedLocs, sortedIn] = sort(locs, 'descend');
% Get the peak (the one around intV intensity)
result = struct();
result.loc = sortedLocs(idx);
result.value = Im(result.loc);
result.p = prominences(sortedIn);
result.p = result.p(idx);
end

function result = findPeakCloserToIntensity(Im, locs, prominences, intImg)
% Find the peak closest to intensity intV
[~, idx] = min(abs(locs - intImg));  
% Get the peak (the one around intV intensity)
result = struct();
result.loc = locs(idx);
result.value = Im(idx);
result.p = prominences(idx);
end

function closestBin = getClosestBin(data, histo)
    [~, closestBin] = min(abs(histo.binCenters - data.value));
end

function logPeak(data, closestBin)
 disp(['Peak intensity:', num2str(data.value), ...
        '-Location', num2str(data.loc), ...
        '-Prominence:', num2str(data.p),...
        '-Closest Bin:', num2str(closestBin)]);
end

function plotHistogramWithLandmarks(subject, histo, landmarks)
    % % Plot the histogram
    figure; % Create a new figure for each plot
    bar(histo.binCenters, histo.counts);  % Plot the histogram
    hold on;
    
    % Mark minI, maxI, pc1, pc2
    h1 = xline(landmarks.min, 'g', 'LineWidth', 2, 'Label', 'minI');
    h2 = xline(landmarks.max, 'b', 'LineWidth', 2, 'Label', 'maxI');
    h3 = xline(landmarks.pc1, 'm', 'LineWidth', 2, 'Label', 'pc1 (10th Percentile)');
    h4 = xline(landmarks.pc2, 'c', 'LineWidth', 2, 'Label', 'pc2 (99.8th Percentile)');

    % Plot multiple peaks
    peakHandles = [];
    for i = 1:length(landmarks.peaks)
        peakHandles(i) = xline(landmarks.peaks{i}.value, 'r', 'LineWidth', 2, ...
            'Label', ['Peak ' num2str(i) ' (Prom: ' num2str(landmarks.peaks{i}.p) ')']);
    end

    % Add labels and title
    title(['Histogram with Landmarks, Subject: ', num2str(subject)]);
    xlabel('Intensity Value');
    ylabel('Count');

    % Create dynamic legend including peaks
    legend([h1, h2, h3, h4, peakHandles], 'Histogram', 'minI', 'maxI', ...
        'pc1', 'pc2', ...
        arrayfun(@(i) ['Peak ' num2str(i)], 1:length(landmarks.peaks), 'UniformOutput', false));

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