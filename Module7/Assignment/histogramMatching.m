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
landmarks = determineLandmarks(imageSet, trainingImageIndices);


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

function landmarks = determineLandmarks(imageSet, trainingImageIndices)
landmarks = {};  
j = 1;
for i = 1:size(imageSet, 2)

    % Read the NIfTI data
    V = imageSet{i}.V;
      
    % Reshape the image matrix into a vector for percentile calculation
    V = V(:);  % Flatten the 3D image volume to 1D vector
    
    if ismember(i, trainingImageIndices)
        minI = min(V);
        maxI = max(V);
        % Calculate pc1 and pc2
        pc1 = prctile(V, 10);
        pc2 = prctile(V, 99.8); 
        % Store the landmarks for this image
        landmarks{j} = struct('min', minI, 'max', maxI,...
        'pc1', pc1, 'pc2', pc2);
    
        % Display percentiles for each image (optional)
        disp(['Image: ', num2str(i), '-minI:', num2str(minI),...
            '-maxI:', num2str(maxI), '- pc1:', num2str(pc1), ...
              '-pc2: ', num2str(pc2)...
              ]);
        j = j + 1;
    end
end

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
    ylabel('Frequency');
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