image = imread('mri.tif'); % Replace with your MRI image path
imshow(image, []);
nBins = 50;
K = histeq(image,nBins);
figure;
imshow(K, []);