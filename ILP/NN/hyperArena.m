%%
clear
clc

addpath(genpath(pwd));

flow=1:10;
NF=length(flow);

layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
load(imgfile);
load(labfile);

%%
training_size=1e3;
batch_size=2e2;
epoch_size=10;
learning_rate=1e-3;
HID_INDEX=4;

imgDataTrain=imgData(:,:,:,1:training_size);
inputSize=size(imgDataTrain);
imgLabelsTrain=categorical(imgLabels(1:training_size,:));

imgDataTest=imgData(:,:,:,7001:7100);
imgLabelsTest=categorical(imgLabels(7001:7100,:));
NUMTEST=size(imgLabelsTest,1);

%%
layer=cell(4,1);
layer{1} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)  

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{2} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
      
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{3} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{4} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layers=layer{HID_INDEX};

options = trainingOptions( 'adam',...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', batch_size,...
    'MaxEpochs', epoch_size, ...
    'InitialLearnRate',learning_rate);

net = cell(NF,1);
trainInfo = cell(NF,1);
training_clock=tic;
for ii=1:NF
    [net{ii},trainInfo{ii}] = trainNetwork(imgDataTrain, imgLabelsTrain(:,ii), layers, options);
end
training_time=toc(training_clock);
training_time=training_time/NF;

training_accuracy=0;
for ii=1:NF
    training_accuracy=training_accuracy+trainInfo{ii}.TrainingAccuracy(end);
end
training_accuracy=training_accuracy/(NF*NUMTEST);

%%
predLabelsTestMedium=cell(NF,1);
score=cell(NF,1);
testing_clock=tic;
for ii=1:NF
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:NF}];
predLabelsTest(isundefined(predLabelsTest))=categorical(1);
scoreTest=[score{1:NF}];
NE=length(layout.image_layout.space.y);
scoreTest(isnan(scoreTest))=1/NE;
testing_time=toc(testing_clock);
testing_time=testing_time/NF;

testing_accuracy=zeros(6,1);
opt.NF=NF;
opt.NT=NUMTEST;
for jj=1:length(testing_accuracy)
    opt.mode=jj;
    testing_accuracy(jj)=ErrorCalc(imgLabelsTest, score, opt);
end

%%
result.training_time=training_time;
result.testing_time=testing_time;
result.training_accuracy=training_accuracy;
result.testing_accuracy=testing_accuracy;

disp(result);
disp(result.testing_accuracy);
