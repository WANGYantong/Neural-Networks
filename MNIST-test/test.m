%%
clear;
clc;

%% load training data and label
imgfile='train-images-idx3-ubyte';
labfile='train-labels-idx1-ubyte';
imgDataTrain=loadMNISTImages(imgfile);
labelsTrain=loadMNISTLabels(labfile);

%% construct network layers
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% training CNN
miniBatchSize = 8192;
options = trainingOptions( 'sgdm',...    % stochastic gradient descent with momentum
    'MiniBatchSize', miniBatchSize,...
    'InitialLearnRate',0.001,...
    'Plots', 'training-progress');

net = trainNetwork(imgDataTrain, labelsTrain, layers, options);	

%% load test data and label
imgfile='t10k-images-idx3-ubyte';
labfile='t10k-labels-idx1-ubyte';
imgDataTest=loadMNISTImages(imgfile);
labelsTest=loadMNISTLabels(labfile);

%% test trained neural network
predLabelsTest = net.classify(imgDataTest);
accuracy = sum(predLabelsTest == labelsTest) / numel(labelsTest);

