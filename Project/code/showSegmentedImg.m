
function showSegmentedImg(Img, segImg, centers, opt)

if ~ isfield(opt, 'alpha')
    opt.alpha = .5;
end

if ~ isfield(opt, 'nClusters')
    opt.nClusters = 4;
end

if ~ isfield(opt, 'fixedColors')
    opt.fixedColors = [
    1 0 0;   % Red for Cluster 1
    0 1 0;   % Green for Cluster 2
    0 0 1;   % Blue for Cluster 3
    1 1 0;   % Yellow for Cluster 4
    ];
end

clusteredImg = label2rgb(segImg, opt.fixedColors);  % Use the fixed color map

h = imshow(clusteredImg);  % Show the segmented image on top
set(h, 'AlphaData', opt.alpha);  % Apply transparency to the overlay
hold off;

title(opt.title, "FontSize", 20, 'FontWeight','bold');

% ----- Adding a Proper Legend with Cluster Colors -----

% Create legend entries for each cluster
legendEntries = cell(opt.nClusters, 1);
for i = 1:opt.nClusters
    legendEntries{i} = sprintf('Cluster %d', i);
end

% Create invisible plots for each cluster color (for the legend)
hold on;
for i = 1:opt.nClusters
    plot(NaN, NaN, 's', 'MarkerSize', 10, ...
        'MarkerFaceColor', opt.fixedColors(i, :), ...
        'MarkerEdgeColor', opt.fixedColors(i, :));
end

% Add the legend with the fixed colors for each cluster
legend(legendEntries, 'Location', 'bestoutside');
hold off;
% 
% % ---------- Mark the Cluster Centers on the Segmented Image ----------
% 
hold on;
nRows = size(Img, 1);
nCols = size(Img, 2);
for i=1:size(centers,1)
    % Convert batCenters from normalized [0, 1] back to the original intensity scale [0, 255]
    scaledCenters = centers * 255;
    colors = opt.centerColors(i,:);
    centerNames = opt.centerNames(i,:);
    % Plot batCenters
    hold on;
    for j = 1:opt.nClusters
        % Find the pixel closest to each bat center
        [~, idx] = min(abs(double(Img(:)) - scaledCenters(i,j)));  % Find closest pixel to the center
        [x, y] = ind2sub([nRows, nCols], idx);  % Convert the 1D index to 2D coordinates (row, col)
    
        % Plot the center as a red cross ('rx') on the segmented MRI image (add to legend)

        plot(y, x, colors, 'MarkerSize', 18, 'LineWidth', 2, 'DisplayName', centerNames);
    end
end

% ----- Add the Legend -----
legend('show', 'Location', 'bestoutside');
hold off;

end