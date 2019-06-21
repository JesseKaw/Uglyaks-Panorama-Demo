% By Jesse from Uglyaks.com
% Design goal:  Similar to my previous coursework, I wanted to create an
%               entirely generic solution for stitching images together and
%               creating a panorama effect. Taking in a configuration file,
%               the user can specify the images to stitch as well as
%               several parameters. So while the solution is generic, the
%               user can improve the quality of the panorama by altering
%               the configuration file.

% Load in the required configuration file
LoadConfig('Landscape');

% Load in the images specified in the file
[Images, NumberOfImages] = LoadImages(ImagesFolder);

% Extract the images and points from the loaded images.
[Points, Features] = GetPointsAndFeatures(Images, NumberOfImages);

% Starting at the first image, set the estimated final image size based on
% the configuration file
Starting = 1;
EstimatedFinalSize = [ReferenceX ReferenceY];

% Stop unimportant warning messages from flooding the console
warning('off','all')

% Stitch the images together starting at the starting image and parameters
% defined in the configuration file
[FinalImage, ComparedImage] = StitchImages(Images, Starting, Points, Features, EstimatedFinalSize, ...
    NumberOfAttempts, Similarity, Length, RequiredMatching);

% Show the stitched image
figure,
imshow(FinalImage);

% FUNCTIONS
function LoadConfig(ConfigFile)

    % Load in the configuation file and split it up
    config = textscan(fopen(cat(2, ConfigFile, '.jc')), "%s %d %d %f %d %f %d");

    % Get the image folder to load
    folder = config(1);
    assignin('base', 'ImagesFolder', cell2mat(folder{1, 1}(1)));    
    assignin('base', 'ReferenceX', cell2mat(config(2)));            % Load in the reference X
    assignin('base', 'ReferenceY', cell2mat(config(3)));            % Load in the reference Y
    assignin('base', 'Similarity', cell2mat(config(4)));            % Get the required similarity
    assignin('base', 'NumberOfAttempts', cell2mat(config(5)));      % Get the number of attempts when stitching
    assignin('base', 'Length', cell2mat(config(6)));                % Get the max lenght between points
    assignin('base', 'RequiredMatching', cell2mat(config(7)));      % Get the minimum number of matching features
end

function [images, numberOfImages] = LoadImages(Folder)
    
    % Get the images from the designated folder
    loadedFolder = dir(Folder);
    
    % Load in all the file names, removing non-files from the results
    images = string();
    images = [images loadedFolder.name];
    images(1:3) = [];
    
    % Append the folder to the beginning of the file names
    images(:) = strcat(Folder, "/", images(:));
    
    % Find the number of images
    numberOfImages = size(images);
    numberOfImages = numberOfImages(2);
    
end

function [points, features] = GetPointsAndFeatures(Images, NumberOfImages)

    % Set up the number of points and features we're going to store
    points = cell(1, NumberOfImages);
    features = cell(1, NumberOfImages);
    
    % Cycle through each image...
    
    for x = 1:NumberOfImages
       
        % Load the image, set to black and while then adjust the contrast
        % for feature detection
        current = imread(char(Images(x)));
        current = rgb2gray(current);
        current = imadjust(current,[0.25 0.55],[]);
        
        % Detect the features and extract them
        currentPoints = detectSURFFeatures(current);        
        [currentPoints, currentFeatures] = extractFeatures(current, currentPoints);
        
        % Add the features to the returning arrays
        points{x} = currentPoints;
        features{x} = currentFeatures;
        
    end

end

function [FinalImage, ComparedImage] = StitchImages(Images, Starting, Points, Features, EstimatedFinalSize, ...
    NumberOfAttempts, Similarity, Length, RequiredMatching)

    % Create a blender to use when stiching images together
    blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
    
    % Create the output view as reference based on the estimated size
    outputView = imref2d([EstimatedFinalSize(2), EstimatedFinalSize(1)]);
    
    % Create the blank starting image
    startingImage = uint8(zeros([EstimatedFinalSize(2), EstimatedFinalSize(1), 3]));
    
    % Load in the image and add it to the blank image, translating it to
    % the middle of the blank image.
    strongImage = imread(char(Images(Starting)));
    mask = imwarp(true(size(strongImage,1),size(strongImage,2)), affine2d([1 0 0; 0 1 0; 0 0 1]), 'OutputView', imref2d(size(strongImage)));
    startingImage = step(blender, startingImage, strongImage, mask);
    startingImage = imtranslate(startingImage, [EstimatedFinalSize(1)/3 EstimatedFinalSize(2)/3]);
    
    % Create an image for feature detection, adjust contrast then extract
    % features
    compareImage = startingImage;
    compareImage = rgb2gray(imadjust(compareImage,[0.25 0.55],[]));
    comparePoints = detectSURFFeatures(compareImage);
    [comparePoints, compareFeatures] = extractFeatures(compareImage, comparePoints);
    
    % Create an array to store all the images we have left to stitch
    imagesToAdd = zeros(size(Images));
    loop = size(imagesToAdd);
    for i = 1:loop(2)
        imagesToAdd(i) = i;
    end
    
    % Get the current image to stitch then start looping
    index = 1;
    currentImage = imagesToAdd(index);
    while(loop(2) ~= 1)
        
        % If we're not trying stitch the starting image to this image...
        if currentImage ~= Starting
            
            % Get the current image's features extracted before
            childPoints = cell2mat(Points(currentImage));
            childFeatures = [Features(1, currentImage)];
            childFeatures = childFeatures{:};
            
            % Match the current image's points against the existing image's
            indexPairs = matchFeatures(comparePoints, childPoints);
            
            % Get the matched features
            matchedCurrent = compareFeatures(indexPairs(:, 1));
            matchedChild = childFeatures(indexPairs(:, 2));
            
            % If the matched features exceed or equal the require set in
            % the configuration file
            matchedSize = size(indexPairs);
            if(matchedSize(1) >= RequiredMatching)
                
                % Remove this image from the images to stitch array
                imagesToAdd(index) = [];
                
                % Get the transform based on configuration parameters
                [tform, ~, ~] = estimateGeometricTransform(...
                    matchedChild, matchedCurrent, 'similarity', 'Confidence', Similarity, 'MaxNumTrials', NumberOfAttempts, 'MaxDistance', Length);
                
                % Read in the current image
                image = imread(char(Images(currentImage)));
                
                % Adjust the contrast for comparison
                compareChildImage = rgb2gray(imadjust(image,[0.25 0.55],[]));
                
                % Create the mask for this image based on the transform
                mask = imwarp(true(size(image,1),size(image,2)), tform, 'OutputView', outputView);
                
                % Transform the image based on the estimated geometric transform
                image = imwarp(image, tform, 'OutputView', outputView);
                compareChildImage = imwarp(compareChildImage, tform, 'OutputView', outputView);

                % Add the images to the final stitched image and compare image
                startingImage = step(blender, startingImage, image, mask);
                compareImage = step(blender, compareImage, compareChildImage, mask);
                
                % Detect and extract features from the new compare image
                comparePoints = detectSURFFeatures(compareImage);
                [comparePoints, compareFeatures] = extractFeatures(compareImage, comparePoints);
                
                % figure, imshow(startingImage);
            else
                
                % Otherwise move onto the next imge
                index = index + 1;
            end
        else
            
            % Otherwize move onto the next image
            index = index + 1;
        end
        
        % Ensure that we're loop is equal to the number of images left to stitch
        loop = size(imagesToAdd);
        
        % If index is greater than the loop, reset it to 1
        if(index > loop(2))
            index = 1;
        end
        
        % Set the next image to stitch to the next index in the stitch array
        currentImage = imagesToAdd(index);
    end
    
    % Find the non-black area for the image we want to return
    [xMin, xMax, yMin, yMax] = GetNonblackArea(startingImage);
    
    % Cut off the black border aound the final and compare image then return them
    FinalImage = startingImage(xMin:xMax, yMin:yMax, :);
    ComparedImage = compareImage(xMin:xMax, yMin:yMax, :);
end

function [xMin, xMax, yMin, yMax] = GetNonblackArea(Image)

    % Scale down the images 10 times for performance
    blackTest = imresize(Image, 0.10, 'nearest');
    nonBlackArea = blackTest ~= 0;
    nonBlackSize = size(nonBlackArea);
    
    % A bit hacky but set the default mins and maxes to change
    xMin = 100000;
    xMax = -1;
    yMin = 100000;
    yMax = -1;
    
    % Loop through the smaller image and find where the populated area of
    % the image is 
    for x = 1:nonBlackSize(1)
        for y = 1:nonBlackSize(2)
            pixel = blackTest(x, y, :);
            
            if(sum(pixel) ~= 0)
                if(x < xMin)
                    xMin = x;
                end
                
                if(x > xMax)
                    xMax = x;
                end
                
                if(y < yMin)
                    yMin = y;
                end
                
                if(y > yMax)
                    yMax = y;
                end
            end
        end
    end
    
    % Undo the image resizing
    xMin = xMin / 0.10;
    xMax = xMax / 0.10;
    yMin = yMin / 0.10;
    yMax = yMax / 0.10;

end

% -- Observations --
%   - The panorama created can be very accurate, however, the performance
%       can suffer. Especially if there are lots of smaller images to
%       stitch together or the resulting image is extremely large.
%   - There might be a way of creating the comparison image when stitching
%       images together by appending the child features and points to the
%       comparison image's. However I didn't have the time to figure out
%       how to do this.
%   - The image seems to work very well with larger fewer images than lots
%       of smaller images. Though I did consider allowing the algorithm to
%       split images up more to allow reduce the number of inaccuracies
%       when estimating the geometric transformation.
%   - Ultimately the starting image doesn't seem to have much of an effect
%       on the final image, hence why it's always the first in the array.
%       By altering the contrast and relying on the SURF feature detection
%       algorithm, it doesn't matter which starting image is selected.
%       Though it would require altering the parameters in the
%       configuration file.