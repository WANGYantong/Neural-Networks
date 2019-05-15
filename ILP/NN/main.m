%%
clear
clc

%% load training/test data and label
imgfile='../DataStore/imgData.mat';
labfile='../DataStore/imgLabels.mat';
load(imgfile);
load(labfile);

imgDataTrain=imgData(:,:,:,1:40000);
imgLabelsTrain=imgLabels(1:40000,:);

imgDataTest=imgData(:,:,:,40001:end);
imgLabelsTest=imgLabels(40001:end,:);

%% construct neural network  layers
layers = [
    imageInputLayer([24 15 1])
	
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
%     maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
%     maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];
%     fullyConnectedLayer(10)
%     regressionLayer];

%% training neural network
miniBatchSize = 1024;
options = trainingOptions( 'sgdm',...  
    'ExecutionEnvironment','cpu',...
    'MiniBatchSize', miniBatchSize,...
    'MaxEpochs', 60, ...
     'InitialLearnRate',0.001...
     );
%     'Plots', 'training-progress');

net = cell(10,1);

parfor ii=1:10
    net{ii} = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,ii)), layers, options);
end
% net=trainNetwork(imgDataTrain, imgLabelsTrain, layers, options);

%% test trained neural network
predLabelsTestMedium = cell(10,1);
parfor ii=1:10
    predLabelsTestMedium{ii} = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:10}];
% predLabelsTest=net.predict(imgDataTest);

counter=0;
for jj=1:length(imgLabelsTest)
    if all(predLabelsTest(jj,:)==categorical(imgLabelsTest(jj,:)))
%     if all(round(predLabelsTest(jj,:))==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy = counter / length(imgLabelsTest);

