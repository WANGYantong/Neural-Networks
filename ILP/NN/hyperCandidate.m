function result=hyperCandidate(training_size,batch_size,epoch_size,learning_rate,HID_INDEX)

flow=1:10;
NF=length(flow);

%% generate training and testing dataset
layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
img=load(imgfile);
lab=load(labfile);

imgDataTrain=img.imgData(:,:,:,1:training_size);
inputSize=size(imgDataTrain);
imgLabelsTrain=categorical(lab.imgLabels(1:training_size,:));

imgDataTest=img.imgData(:,:,:,7001:7100);
imgLabelsTest=categorical(lab.imgLabels(7001:7100,:));
NUMTEST=size(imgLabelsTest,1);

%% CNN structure
layer=cell(10,1);
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

layer{16} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layers=layer{HID_INDEX};

options = trainingOptions( 'adam',...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', batch_size,...
    'MaxEpochs', epoch_size, ...
    'InitialLearnRate',learning_rate,...
    'Verbose',false);

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

predLabelsTestMedium=cell(NF,1);
score=cell(NF,1);
testing_clock=tic;
for ii=1:NF
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
% predLabelsTest=[predLabelsTestMedium{1:NF}];
% predLabelsTest(isundefined(predLabelsTest))=categorical(1);
% scoreTest=[score{1:NF}];
% NE=length(layout.image_layout.space.y);
% scoreTest(isnan(scoreTest))=1/NE;
testing_time=toc(testing_clock);
testing_time=testing_time/NF;

testing_accuracy=zeros(6,1);
opt.NF=NF;
opt.NT=NUMTEST;
for jj=1:length(testing_accuracy)
    opt.mode=jj;
    testing_accuracy(jj)=ErrorCalc(imgLabelsTest, score, opt);
end

result.training_time=training_time;
result.testing_time=testing_time;
result.training_accuracy=training_accuracy;
result.testing_accuracy=testing_accuracy;

fprintf('\n/**************************************************/\n');
fprintf('training_size=%d,batch_size=%d,\nepoch_size=%d,learning_rate=%.4f,hidden_layer=%d\n',...
    training_size,batch_size,epoch_size,learning_rate,HID_INDEX);
% fprintf('\n training time:%.4f',result.training_time);
% fprintf('\n testing time:%.4f',result.testing_time);
% fprintf('\n training accuracy:%.5f',result.training_accuracy);
% fprintf('\n testing accuracy(e1):%.5f',result.testing_accuracy(1));
% fprintf('\n testing accuracy(e2):%.5f',result.testing_accuracy(2));
% fprintf('\n testing accuracy(e3):%.5f',result.testing_accuracy(3));
% fprintf('\n testing accuracy(e4):%.5f',result.testing_accuracy(4));
% fprintf('\n testing accuracy(e5):%.5f',result.testing_accuracy(5));
fprintf('\n');

end

