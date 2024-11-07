dataDir = "/Users/yvesgreatti/github/AppliedMedicalImageProcessing/Project/data/";
imageDir = fullfile(dataDir,"BraTS");
filename = matlab.internal.examples.downloadSupportFile(...
    "vision","data/sampleBraTSTestSetValid.tar.gz");
untar(filename,imageDir);

trainDataFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","imagesTest","BraTS447.mat");
testDataFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","imagesTest","BraTS463.mat");
testLabelFileName = fullfile(imageDir,...
    "sampleBraTSTestSetValid","labelsTest","BraTS463.mat");
%%
openExample('fuzzy/BrainTumorSegmentationUsingFuzzyCMeansClusteringExample')
%%
function y = createMovingWindowFeatures(in,dim)
% Create feature vectors using a moving window.

rStep = floor(dim(1)/2);
cStep = floor(dim(2)/2);

x1 = [zeros(size(in,1),rStep) in zeros(size(in,1),rStep)];
x = [zeros(cStep,size(x1,2));x1;zeros(cStep,size(x1,2))];

[row,col] = size(x);
yCol = prod(dim);
y = zeros((row-2*rStep)*(col-2*cStep), yCol);
ct = 0;
for rId = rStep+1:row-rStep
    for cId = cStep+1:col-cStep
        ct = ct + 1;
        y(ct,:) = reshape(x(rId-rStep:rId+rStep,cId-cStep:cId+cStep),1,[]);
    end
end
end

function [hasTumor,numFalsePos,tumorLabel] = ...
    segmentTumor(testLabel,refPositiveIds,clusterId)
%% Calculate detection results using the test and reference data.

tumorIds = testLabel==clusterId;
segmentedImage = testLabel;
segmentedImage(tumorIds) = 1;
segmentedImage(~tumorIds) = 0;
tumorIdsECIds = find(tumorIds==1);
hasTumor = ~isempty(tumorIdsECIds);
numFalsePos = length(find(setdiff(tumorIdsECIds,refPositiveIds)));
tumorLabel = segmentedImage;
end