function result=hyperCandidate(training_size,batch_size,epoch_size,learning_rate,HID_INDEX)

flow=1:5;
NF=length(flow);

%% generate training and testing dataset
layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
img=load(imgfile);
lab=load(labfile);

off_load=0;
testing_range=off_load+1:off_load+128;
imgDataTest=img.imgData(:,:,:,testing_range);
imgLabelsTest=categorical(lab.imgLabels(testing_range,:));
NUMTEST=length(testing_range);

off_load=testing_range(end);
validation_range=off_load+1:off_load+128;
imgDataValid=img.imgData(:,:,:,validation_range);
imgLabelsValid=categorical(lab.imgLabels(validation_range,:));

off_load=validation_range(end);
training_range=off_load+1:off_load+training_size;
imgDataTrain=img.imgData(:,:,:,training_range);
inputSize=size(imgDataTrain);
imgLabelsTrain=categorical(lab.imgLabels(training_range,:));

%% CNN structure
layer=cell(16,1);
layer{1} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)  

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{2} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
      
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{3} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
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
%     maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{4} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
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
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{5} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
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
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,128,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

layer{16} = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
%     maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,8,'Padding','same')
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

for ii=1:NF
    options(ii) = trainingOptions( 'sgdm',...
        'ExecutionEnvironment','auto',...
        'MiniBatchSize', batch_size,...
        'MaxEpochs', epoch_size, ...
        'Shuffle','every-epoch',...
        'InitialLearnRate',learning_rate,...
        'L2Regularization',0.0005,...
        'ValidationData',{imgDataValid,imgLabelsValid(:,ii)},...
        'ValidationFrequency',16,...
        'Verbose',false);
%         'Plots','training-progress');
end

net = cell(NF,1);
trainInfo = cell(NF,1);
training_clock=tic;
for ii=1:NF
    [net{ii},trainInfo{ii}] = trainNetwork(imgDataTrain, imgLabelsTrain(:,ii), layers, options(ii));
end
training_time=toc(training_clock);
training_time=training_time/NF;

training_accuracy=0;
training_loss=0;
validation_accuracy=0;
validation_loss=0;
for ii=1:NF
    training_accuracy=training_accuracy+trainInfo{ii}.TrainingAccuracy(end);
    training_loss=training_loss+trainInfo{ii}.TrainingLoss(end);
    validation_accuracy=validation_accuracy+trainInfo{ii}.ValidationAccuracy(end);
    validation_loss=validation_loss+trainInfo{ii}.ValidationLoss(end);
end
training_accuracy=training_accuracy/(NF*100);
training_loss=training_loss/NF;
validation_accuracy=validation_accuracy/(NF*100);
validation_loss=validation_loss/NF;

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
result.training_loss=training_loss;
result.validation_accuracy=validation_accuracy;
result.validation_loss=validation_loss;
result.testing_accuracy=testing_accuracy;

fprintf('\n/**************************************************/\n');
fprintf('training_size=%d,batch_size=%d,\nepoch_size=%d,learning_rate=%.4f,hidden_layer=%d\n',...
    training_size,batch_size,epoch_size,learning_rate,HID_INDEX);
% fprintf('training_accuracy=%.4f,training_loss=%.4f,\nvalidation_accuracy=%.4f,validation_loss=%.4f\n',...
%     training_accuracy,training_loss,validation_accuracy,validation_loss);
% fprintf('testing_accuracy=%.4f,training_time=%.4f\n',testing_accuracy(1), training_time);
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

