
function filteredI = BilateralFilterGraySep(I, sigmaD, sigmaR)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigmaD  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigmaR  - standard deviation for the range Gaussian kernel
    % Output:
    %   filteredI - output filtered grayscale image of size M x N
    
    % Convert image to double for precision
    I = double(I);
    
    % Get the size of the image
    [M, N] = size(I);
    
    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigmaD);
    
    % First pass: horizontal filtering
    I_horizontal = zeros(M, N); % This will store the intermediate result
    padded_I = padarray(I, [K 0], 'symmetric'); % Pad only along rows for horizontal pass
    
    % Horizontal pass
    for u = 1:M
        for v = 1:N
            % Initialize accumulators
            S = 0;
            W = 0;
            
            % Get the intensity of the center pixel
            a = I(u, v);
            
            % Loop over the horizontal neighborhood
            for m = -K:K
                % Get the neighboring pixel (horizontal axis)
                b = padded_I(u + m + K, v); % +K offset to handle padding indexing
                
                % Compute spatial (domain) Gaussian weight
                wd = exp(-m^2 / (2 * sigmaD^2));
                
                % Compute range Gaussian weight
                wr = exp(-(a - b)^2 / (2 * sigmaR^2));
                
                % Total weight
                w = wd * wr;
                
                % Update accumulators
                S = S + w * b;
                W = W + w;
            end
            
            % Set the intermediate filtered value for the horizontal pass
            I_horizontal(u, v) = S / W;
        end
    end
    
    % Second pass: vertical filtering
    filteredI = zeros(M, N); % This will store the final result
    padded_I_horizontal = padarray(I_horizontal, [0 K], 'symmetric'); % Pad only along columns
    
    % Vertical pass
    for u = 1:M
        for v = 1:N
            % Initialize accumulators
            S = 0;
            W = 0;
            
            % Get the intensity of the center pixel in the intermediate result
            a = I_horizontal(u, v);
            
            % Loop over the vertical neighborhood
            for n = -K:K
                % Get the neighboring pixel (vertical axis)
                b = padded_I_horizontal(u, v + n + K); % +K offset to handle padding indexing
                
                % Compute spatial (domain) Gaussian weight
                wd = exp(-n^2 / (2 * sigmaD^2));
                
                % Compute range Gaussian weight
                wr = exp(-(a - b)^2 / (2 * sigmaR^2));
                
                % Total weight
                w = wd * wr;
                
                % Update accumulators
                S = S + w * b;
                W = W + w;
            end
            
            % Set the final filtered value for the vertical pass
            filteredI(u, v) = S / W;
        end
    end
end


function filteredI = BilateralFilterGray(I, sigmaD, sigmaR)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigmaD  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigma_r  - standard deviation for the range Gaussian kernel
    % Output:
    %   filteredI - output filtered grayscale image of size M x N
    
    % Get the size of the image
    [M, N] = size(I);
    
    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigmaD);
    
    % Initialize the output image as a copy of the input image
    filteredI = I;
    
    % For each pixel in the image
    for u = 1:M
        for v = 1:N
            % Initialize weight sums
            S = 0;
            W = 0;
            
            % Get the intensity of the center pixel
            a = I(u, v);
            
            % Loop over the neighborhood of the current pixel
            for m = -K:K
                for n = -K:K
                    % Check if the neighboring pixel is within bounds
                    if u + m > 0 && u + m <= M && v + n > 0 && v + n <= N
                        % Get the intensity of the neighboring pixel
                        b = I(u + m, v + n);
                        
                        % Compute the spatial (domain) Gaussian weight
                        wd = exp(-(m^2 + n^2) / (2 * sigmaD^2));
                        
                        % Compute the range Gaussian weight
                        wr = exp(-((a - b)^2) / (2 * sigmaR^2));
                        
                        % Total weight
                        w = wd * wr;
                        
                        % Update weights sums
                        S = S + w * b;
                        W = W + w;
                    end
                end
            end
            
            % Set the new pixel value
            filteredI(u, v) = S / W;
        end
    end
end

function I_filtered = BilateralFilterGrayOptimized(I, sigma_d, sigma_r)
    % Input:
    %   I        - input grayscale image of size M x N
    %   sigma_d  - standard deviation for the domain (spatial) Gaussian kernel
    %   sigma_r  - standard deviation for the range Gaussian kernel
    % Output:
    %   I_filtered - output filtered grayscale image of size M x N
    
    % Convert image to double for precision
    I = double(I);
    
    % Get the size of the image
    [M, N] = size(I);
    
    % Compute the spatial Gaussian kernel size K
    K = ceil(3.5 * sigma_d);
    
    % Create a grid of distances (m^2 + n^2) for the spatial Gaussian (domain kernel)
    [X, Y] = meshgrid(-K:K, -K:K);
    spatial_weights = exp(-(X.^2 + Y.^2) / (2 * sigma_d^2));
    
    % Initialize the output image
    I_filtered = zeros(M, N);
    
    % Pad the image to handle borders
    padded_I = padarray(I, [K K], 'symmetric');
    
    % For each pixel in the image
    for u = 1:M
        for v = 1:N
            % Extract the local neighborhood around pixel (u,v)
            local_region = padded_I(u:u+2*K, v:v+2*K);
            
            % Compute the range Gaussian weights based on intensity differences
            intensity_diff = local_region - I(u,v);
            range_weights = exp(-(intensity_diff.^2) / (2 * sigma_r^2));
            
            % Compute the combined bilateral weights
            bilateral_weights = spatial_weights .* range_weights;
            
            % Normalize the weights
            normalized_weights = bilateral_weights / sum(bilateral_weights(:));
            
            % Compute the filtered intensity as the weighted sum of the local neighborhood
            I_filtered(u, v) = sum(sum(normalized_weights .* local_region));
        end
    end
end
