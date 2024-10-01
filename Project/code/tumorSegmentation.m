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