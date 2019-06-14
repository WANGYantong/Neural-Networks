%%
clear
clc

%% load training/test data and label
imgfile='../DataStore/imgData.mat';
labfile='../DataStore/imgLabels.mat';
load(imgfile);
load(labfile);

imgDataTrain=imgData(:,:,:,1:8000);
imgLabelsTrain=imgLabels(1:8000,:);

%
% imgLabelsTrain2=zeros(8000,5);
% for ii=1:5
%     resu=strcat(num2str(imgLabelsTrain(:,2*ii-1)),num2str(imgLabelsTrain(:,2*ii)));
%     imgLabelsTrain2(:,ii)=str2num(resu);
% end
    
imgDataTest=imgData(:,:,:,8001:10000);
imgLabelsTest=imgLabels(8001:10000,:);

%
% imgLabelsTest2=zeros(2000,5);
% for ii=1:5
%     resu=strcat(num2str(imgLabelsTest(:,2*ii-1)),num2str(imgLabelsTest(:,2*ii)));
%     imgLabelsTest2(:,ii)=str2num(resu);
% end

%% construct neural network  layers
layers = [
    imageInputLayer([16 11 1]) % sparse 24*15, dense 21*10
	
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
options = trainingOptions( 'adam',...  
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', miniBatchSize,...
    'MaxEpochs', 40, ...
     'InitialLearnRate',0.001,...
    'Plots', 'training-progress');

net = cell(10,1);

for ii=1:10
    net{ii} = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,ii)), layers, options);
end
% net=trainNetwork(imgDataTrain, imgLabelsTrain, layers, options);

%% test trained neural network
predLabelsTestMedium = cell(10,1);
score = cell(10,1);
for ii=1:10
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:10}];
scoreTest=[score{1:10}];
parfor ii=1:2000
    predLabelsTest(ii,:)=combiner(imgDataTest(:,:,:,ii),predLabelsTest(ii,:), scoreTest(ii,:),2);
end
% predLabelsTest=net.predict(imgDataTest);

counter=0;
for jj=1:length(imgLabelsTest)
    if all(predLabelsTest(jj,:)==categorical(imgLabelsTest(jj,:)))
%     if all(round(predLabelsTest(jj,:))==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_final = counter / length(imgLabelsTest);

accuracy=zeros(1,10);
for ii=1:10
    accuracy(ii) = sum(predLabelsTestMedium{ii} ==categorical(imgLabelsTest(:,ii))) / length(imgLabelsTest);
end