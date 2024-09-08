% Load the MRI data provided by MathWorks
load mri;

% Select slice 15 and 2- from the MRI data
slice15 = D(:,:, 1, 15);
slice20 = D(:,:, 1, 20);

% 1. Perform the Radon transform on slice 15 using angles from 0 to 170
% degrees.
angles = 0:179;
[R15, ~] = radon(slice15, angles);
[R20, xp] = radon(slice20, angles);

% 2. Perform 1D Fourier transform on Radon-transformed signal
% from slice 15 at 0 and 90 degrees

% Radon transforms at 0 and 90 degrees.
R15_0 = R15(:, angles == 0);   
R15_90 = R15(:, angles == 90); 
R20_0 = R20(:, angles == 0);   
R20_90 = R20(:, angles == 90); 

% 1D Fourier Transform on Radon transformed signals at 0 and 90 degrees.
% We take fft followed by fftshift to rearrange the output of the fft so
% that the zero frequency component moves to the center of the array.

FT1D_R15_0 = fftshift(fft(R15_0));
FT1D_R15_90 = fftshift(fft(R15_90));
FT1D_R20_0 = fftshift(fft(R20_0));
FT1D_R20_90 = fftshift(fft(R20_90));

% 3. Perform 2D Fourier transform on slice 15 and 20
FT2DSlice15 = fftshift(fft2(slice15));
FT2DSlice20 = fftshift(fft2(slice20));

% 4. Compare the direct and projection-slice Fourier transforms 
% for the two angles using slice 15. 
% For comparison, we use the magnitude signals and plot them on top of each other.

% Dividing 2D Fourier transform by 2 and adding 1 locates the zero frequency component 
% along the y-axis (second dimension). 
% This central column contains the Fourier transform coefficients 
% for the vertical direction (0 Degrees in the spatial domain).
FT2D_Slice_15_0 = FT2DSlice15(:, size(FT2DSlice15, 2) / 2 + 1); 
FT2D_Slice_20_0 = FT2DSlice20(:, size(FT2DSlice20, 2) / 2 + 1); 

% Similarly, we extract Fourier transform coefficients of 
% the central horizontal line along the x-axis (90 Degrees)
FT2D_Slice_15_90 = FT2DSlice15(size(FT2DSlice15, 1) / 2 + 1, :); 
FT2D_Slice_20_90 = FT2DSlice20(size(FT2DSlice20, 1) / 2 + 1, :); 

% Plot comparisons
figure;

% Comparison for 0 degrees
subplot(1, 2, 1);
plot(linspace(min(xp), max(xp), length(FT2D_Slice_15_0)),...
abs(FT2D_Slice_15_0), 'b', 'LineWidth', 1.5, 'DisplayName', '2D FT');
hold on;
plot(xp, abs(FT1D_R15_0), 'r', 'LineWidth', 1.5, 'DisplayName', 'Radon FT');
hold on;
title('Fourier Transforms at 0 Degrees');
xlabel('Frequency');
ylabel('Magnitude');
legend;
grid on;

% Comparison for 90 degrees
subplot(1, 2, 2);
plot(linspace(min(xp), max(xp), length(FT2D_Slice_15_90)),...
    abs(FT2D_Slice_15_90), 'b', 'LineWidth', 1.5, 'DisplayName', '2D FT');
hold on;
plot(xp, abs(FT1D_R15_90), 'r', 'LineWidth', 1.5, 'DisplayName', 'Radon FT ');
hold on;
title('Fourier Transforms at 90 Degrees');
xlabel('Frequency');
ylabel('Magnitude');
legend;
grid on;

% 5. We use the 1D Fourier transform on the Radon-transformed signal 
% from slice 15 (for angles 0 and 90 degrees) and 
% compare it with the projection-slice Fourier transform of slice 20
% (using the same angles and magnitude signal). 
% Plot the results and compare them with previous questions.

% Plot comparisons
figure;

% Comparison for 0 degrees
subplot(1, 2, 1);
plot(xp, abs(FT1D_R15_0), 'r', 'LineWidth', 1.5, 'DisplayName', 'Radon FT at Slice 15');
hold on;
plot(linspace(min(xp), max(xp), length(FT2D_Slice_20_0)),...
    abs(FT2D_Slice_20_0), 'g', 'LineWidth', 1.5, 'DisplayName', 'Radon FT at Slice 20');
title('Fourier Transforms at 0 Degree of Radon-transformed Signals');
xlabel('Frequency');
ylabel('Magnitude');
legend;
grid on;

% Comparison for 90 degrees
subplot(1, 2, 2);
plot(xp, abs(FT1D_R15_90), 'r', 'LineWidth', 1.5, 'DisplayName', 'Radon FT at Slice 15');
hold on;
plot(linspace(min(xp), max(xp), length(FT2D_Slice_20_90)),...
    abs(FT2D_Slice_20_90), 'g', 'LineWidth', 1.5, 'DisplayName', 'Radon FT at Slice 20');
title('Fourier Transforms at 90 Degrees of Radon-transformed Signals');
xlabel('Frequency');
ylabel('Magnitude');
legend;
grid on;