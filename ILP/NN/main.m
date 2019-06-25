%%
clear
clc

%% load training/test data and label
imgfile='../DataStore/imgData.mat';
labfile='../DataStore/imgLabels.mat';
load(imgfile);
load(labfile);

imgDataTrain=imgData(:,:,:,1:8000);
inputSize=size(imgDataTrain);
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
    
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];
%     fullyConnectedLayer(10)
%     regressionLayer];

%% training neural network
miniBatchSize = 256;
options = trainingOptions( 'adam',...  
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', miniBatchSize,...
    'MaxEpochs', 40, ...
     'InitialLearnRate',0.001);%,...
%     'Plots', 'training-progress');

net = cell(10,1);

parfor ii=1:10
    net{ii} = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,ii)), layers, options);
end
% net=trainNetwork(imgDataTrain, imgLabelsTrain, layers, options);
save('net.mat','net');
%% test trained neural network
predLabelsTestMedium = cell(10,1);
score = cell(10,1);
for ii=1:10
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:10}];
scoreTest=[score{1:10}];

NUMTEST=size(imgLabelsTest,1);
parfor ii=1:NUMTEST
    predLabelsTest(ii,:)=combiner(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:),2);
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

% counter__=zeros(NUMTEST,1);
% for ii=1:NUMTEST
%     counter__(ii)=sum(predLabelsTest(ii,:)==categorical(imgLabelsTest(ii,:)));
% end
% result=zeros(10,1);
% for ii=10:-1:1
%     result(11-ii)=sum(counter__==ii);
% end
value__=zeros(NUMTEST,1);
for ii=1:NUMTEST
    value__(ii)=valueCalculator(imgDataTest(:,:,:,ii),predLabelsTest(ii,:));
end