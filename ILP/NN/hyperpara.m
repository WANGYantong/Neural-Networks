%%
clear
clc

addpath(genpath(pwd));

%% load training/test data and label
layout=load('../DataStore/layout.mat');
imgfile=['../DataStore/imgData_' num2str(layout.image_layout.opts) '.mat'];
labfile=['../DataStore/imgLabels_' num2str(layout.image_layout.opts) '.mat'];
load(imgfile);
load(labfile);

imgDataTrain=imgData(:,:,:,1:8000);
inputSize=size(imgDataTrain);
imgLabelsTrain=imgLabels(1:8000,:);

imgDataTest=imgData(:,:,:,8001:10000);
imgLabelsTest=imgLabels(8001:10000,:);

%% construct neural network layers
layers = [
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(6)
    softmaxLayer
    classificationLayer];

%% training
miniBatchSize = [64,128,256,512,1024];
NB=length(miniBatchSize);
options=cell(NB,1);
for ii=1:NB
    options{ii} = trainingOptions( 'adam',...
        'ExecutionEnvironment','auto',...
        'MiniBatchSize', miniBatchSize(ii),...
        'MaxEpochs', 20, ...
        'InitialLearnRate',0.001);
end

net = cell(NB,10);
trainInfo = cell(NB,10);

for ii=1:NB
    parfor jj=1:10
        [net{ii,jj},trainInfo{ii,jj}] = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,jj)), layers, options{ii});
    end
end

trainAccu=zeros(NB,40);
gap=[125,62,31,15,7];
for ii=1:NB
    for jj=1:20
        counter=0;
        for kk=1:10
            counter=counter+trainInfo{ii,kk}.TrainingAccuracy(1+gap(ii)*(jj-1));
        end
        trainAccu(ii,jj)=counter/10;
    end
end

figure;
hold on;
plot([1:20],trainAccu(1,:));
plot([1:20],trainAccu(2,:));
plot([1:20],trainAccu(3,:));
plot([1:20],trainAccu(4,:));
plot([1:20],trainAccu(5,:));
hold off;

%% testing
accuracy_final=zeros(NB,1);
precision=zeros(NB,1);
total_cost=zeros(NB,1);

for ii=1:NB
    
    predLabelsTestMedium = cell(10,1);
    score = cell(10,1);
    
    for jj=1:10
        [predLabelsTestMedium{jj}, score{jj}] = net{ii,jj}.classify(imgDataTest);
    end
    predLabelsTest=[predLabelsTestMedium{1:10}];
    scoreTest=[score{1:10}];
    
    NUMTEST=size(imgLabelsTest,1);
    parfor jj=1:NUMTEST
        predLabelsTest(jj,:)=combiner_II(imgDataTest(:,:,:,jj), predLabelsTest(jj,:), scoreTest(jj,:));
    end
    
    counter=0;
    for jj=1:length(imgLabelsTest)
        if all(predLabelsTest(jj,:)==categorical(imgLabelsTest(jj,:)))
            counter=counter+1;
        end
    end
    accuracy_final(ii) = counter / length(imgLabelsTest);
    
    counter__=zeros(NUMTEST,1);
    for jj=1:NUMTEST
        counter__(jj)=sum(predLabelsTest(jj,:)==categorical(imgLabelsTest(jj,:)));
    end
    result=zeros(10,1);
    for jj=1:11
        result(jj)=sum(counter__==jj-1);
    end
    
    precision(ii)=[0:10]*result/20000;
    
    value__=zeros(NUMTEST,1);
    opt.mode=0;
    for jj=1:NUMTEST
        value__(jj)=valueCalculator(imgDataTest(:,:,:,jj),predLabelsTest(jj,:),opt);
    end
    
    total_cost(ii)=mean(value__);
    
end

save('hyperpara.mat');