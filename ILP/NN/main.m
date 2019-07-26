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

% imshow(imgDataTrain(:,:,:,1),[0,255],'Border','tight','initialMagnification','fit');

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
trainInfo = cell(10,1);

parfor ii=1:10
    [net{ii},trainInfo{ii}] = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,ii)), layers, options);
end
% net=trainNetwork(imgDataTrain, imgLabelsTrain, layers, options);
save(['net_' num2str(layout.image_layout.opts) '.mat'],'net');
%% test trained neural network
predLabelsTestMedium = cell(10,1);
score = cell(10,1);
% tic;
for ii=1:10
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:10}];
scoreTest=[score{1:10}];

NUMTEST=size(imgLabelsTest,1);
parfor ii=1:NUMTEST
%     predLabelsTest(ii,:)=combiner(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:), 2);
    predLabelsTest(ii,:)=combiner_II(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
end
% predLabelsTest=net.predict(imgDataTest);
% running_time=toc;

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

counter__=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter__(ii)=sum(predLabelsTest(ii,:)==categorical(imgLabelsTest(ii,:)));
end
result=zeros(10,1);
for ii=1:11
    result(ii)=sum(counter__==ii-1);
end

precision=[0:10]*result/20000;

value__=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value__(ii)=valueCalculator(imgDataTest(:,:,:,ii),predLabelsTest(ii,:),opt);
end

value=zeros(NUMTEST,1);
opt.mode=1;
offload=inputSize(end);
sol=load('../DataStore/solutions.mat');
for ii=1:NUMTEST
    opt.y=sol.result{ii+offload}.sol.y;
    opt.z=sol.result{ii+offload}.sol.z;
    value(ii)=valueCalculator(imgDataTest(:,:,:,ii),imgLabelsTest(ii,:),opt);
end

%% Greedy
solution_G=zeros(size(imgLabelsTest));
parfor ii=1:NUMTEST
    solution_G(ii,:)=Greedy(imgDataTest(:,:,:,ii));
end

counter=0;
for jj=1:length(imgLabelsTest)
    if all(solution_G(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_Greedy = counter / length(imgLabelsTest);

counter_Greedy=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_Greedy(ii)=sum(solution_G(ii,:)==imgLabelsTest(ii,:));
end
result_Greedy=zeros(10,1);
for ii=1:11
    result_Greedy(ii)=sum(counter_Greedy==ii-1);
end

precision_Greedy=[0:10]*result_Greedy/20000;

value_Greedy=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_Greedy(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_G(ii,:),opt);
end

%% test randomized as comparison
solution_R=zeros(size(imgLabelsTest));
parfor ii=1:NUMTEST
    solution_R(ii,:)=Randomized(imgDataTest(:,:,:,ii));
end

counter=0;
for jj=1:length(imgLabelsTest)
    if all(solution_R(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_Random = counter / length(imgLabelsTest);

counter_Random=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_Random(ii)=sum(solution_R(ii,:)==imgLabelsTest(ii,:));
end
result_Random=zeros(10,1);
for ii=1:11
    result_Random(ii)=sum(counter_Random==ii-1);
end

precision_Random=[0:10]*result_Random/20000;

value_Random=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_Random(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_R(ii,:),opt);
end

save('July23.mat');