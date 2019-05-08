function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fid = fopen(filename, 'rb');
assert(fid ~= -1, ['Could not open ', filename, '']);

magic = fread(fid, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');

labels = categorical(fread(fid, numLabels, 'uint8'));

fclose(fid);

end


