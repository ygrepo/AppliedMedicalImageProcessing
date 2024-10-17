function h = computeHisto(K, I, B)
    % K: Maximum possible intensity value + 1 (e.g., 256 for 8-bit images)
    % I: Intensity array (1D or 2D)
    % B: Number of bins
    
    % Normalize the intensities to range [0, 1]
    I = double(I) / (K - 1); % Assuming K-1 is the maximum intensity value
    
    % Initialize histogram array with zeros for B bins
    h = zeros(1, B); 
    
    % Define bin edges
    bin_edges = linspace(0, 1, B+1); % Create B bins from 0 to 1
    
    % Iterate through the intensity values and count occurrences into bins
    for idx = 1:length(I(:))
        intensity_value = I(idx); % Get the normalized intensity value (range 0-1)
        
        % Find which bin this intensity belongs to
        for bin_idx = 1:B
            if intensity_value >= bin_edges(bin_idx) && intensity_value < bin_edges(bin_idx+1)
                h(bin_idx) = h(bin_idx) + 1; % Increment the count for that bin
                break;
            end
        end
    end
    
    % To handle the case where an intensity equals the maximum value (which falls
    % into the last bin):
    h(B) = h(B) + sum(I(:) == 1);
end
