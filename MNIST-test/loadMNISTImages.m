function imgDataTrain = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fid = fopen(filename, 'rb');
assert(fid ~= -1, ['Could not open ', filename, '']);

magic = fread(fid, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImgs = fread(fid, 1, 'int32', 0, 'ieee-be');
numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
numCols = fread(fid, 1, 'int32', 0, 'ieee-be');

rawImgDataTrain = uint8 (fread(fid, numImgs * numRows * numCols, 'uint8'));

% Reshape the data part into a 4D array
rawImgDataTrain = reshape(rawImgDataTrain, [numRows, numCols, numImgs]);
imgDataTrain(:,:,1,:) = uint8(rawImgDataTrain(:,:,:));

fclose(fid);

end


