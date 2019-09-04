%%
clear
clc

addpath(genpath(pwd));

GREEDY=1;
RANDOM=1;
TR_Ratio=0.8;
%% load training/test data and label
global flow;
NF=length(flow);

layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
load(imgfile);
load(labfile);

TOTAL=size(imgLabels,1);
imgDataTrain=imgData(:,:,:,1:floor(TR_Ratio*TOTAL));
inputSize=size(imgDataTrain);
imgLabelsTrain=imgLabels(1:floor(TR_Ratio*TOTAL),:);

% imshow(imgDataTrain(:,:,:,1),[0,255],'Border','tight','initialMagnification','fit');

%
% imgLabelsTrain2=zeros(8000,5);
% for ii=1:5
%     resu=strcat(num2str(imgLabelsTrain(:,2*ii-1)),num2str(imgLabelsTrain(:,2*ii)));
%     imgLabelsTrain2(:,ii)=str2num(resu);
% end
if NF==5
    imgDataTest=imgData(:,:,:,(TR_Ratio*TOTAL+1):TOTAL);
    imgLabelsTest=imgLabels((TR_Ratio*TOTAL+1):TOTAL,:);
    
    NUMTEST=size(imgLabelsTest,1);
else
    imgDataTest=imgData;
    imgLabelsTest=imgLabels;
    NUMTEST=size(imgLabelsTest,1);
end
%
% imgLabelsTest2=zeros(2000,5);
% for ii=1:5
%     resu=strcat(num2str(imgLabelsTest(:,2*ii-1)),num2str(imgLabelsTest(:,2*ii)));
%     imgLabelsTest2(:,ii)=str2num(resu);
% end

%% train/load CNN
if NF==5
    % construct neural network  layers
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
        
        %     convolution2dLayer(3,128,'Padding','same')
        %     batchNormalizationLayer
        %     reluLayer
        
        fullyConnectedLayer(numel(unique(imgLabelsTrain)))
        softmaxLayer
        classificationLayer];
    %     fullyConnectedLayer(10)
    %     regressionLayer];
    
    %% training neural network
    miniBatchSize = 256;
    options = trainingOptions( 'adam',...
        'ExecutionEnvironment','auto',...
        'MiniBatchSize', miniBatchSize,...
        'MaxEpochs', 30, ...
        'InitialLearnRate',0.001,...
        'Plots', 'training-progress');
    
    net = cell(NF,1);
    trainInfo = cell(NF,1);
    
    for ii=1:NF
        [net{ii},trainInfo{ii}] = trainNetwork(imgDataTrain, categorical(imgLabelsTrain(:,ii)), layers, options);
    end
    % net=trainNetwork(imgDataTrain, imgLabelsTrain, layers, options);
    save(['net_',num2str(layout.image_layout.opts),'_flow',num2str(NF), '.mat'],'net');
else
    load('net_4_flow5.mat');
end
%% test trained neural network
NI=NF/size(net,1); % number of iterations, must be an integer
predLabelsTestMedium = cell(NF,1);
score = cell(NF,1);
imgDataCopy=imgDataTest;
% tic;
for ii=1:NI
    for jj=1:size(net,1)
        index=size(net,1)*(ii-1)+jj; % index of flows
        [predLabelsTestMedium{index}, score{index}] = net{jj}.classify(imgDataCopy(5*(ii-1)+1:5*(ii-1)+size(net,1),:,:,:));
    end
    %update image
    imgDataCopy=imgUpdate(imgDataTest,predLabelsTestMedium);
end
predLabelsTest=[predLabelsTestMedium{1:NF}];
scoreTest=[score{1:NF}];

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

accuracy=zeros(1,NF);
for ii=1:NF
    accuracy(ii) = sum(predLabelsTestMedium{ii} ==categorical(imgLabelsTest(:,ii))) / length(imgLabelsTest);
end

counter__=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter__(ii)=sum(predLabelsTest(ii,:)==categorical(imgLabelsTest(ii,:)));
end
result=zeros(NF,1);
for ii=1:(NF+1)
    result(ii)=sum(counter__==ii-1);
end

precision=(0:NF)*result/(NF*NUMTEST);

value__=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value__(ii)=valueCalculator(imgDataTest(:,:,:,ii),predLabelsTest(ii,:),opt);
end

value=zeros(NUMTEST,1);
opt.mode=1;
if NF==5
    offload=inputSize(end);
else
    offload=0;
end
sol=load(['../DataStore/flow',num2str(flow(end)),'/solutions.mat']);
for ii=1:NUMTEST
%     opt.y=sol.result{ii+offload}.sol.y;
%     opt.z=sol.result{ii+offload}.sol.z;
%     value(ii)=valueCalculator(imgDataTest(:,:,:,ii),imgLabelsTest(ii,:),opt);
    % or using the fval from MILP solver
    value(ii)=sol.result{ii+offload}.fval;
end

%% Greedy
if GREEDY

tic;
solution_G=zeros(size(imgLabelsTest));
for ii=1:NUMTEST
    solution_G(ii,:)=Greedy(imgDataTest(:,:,:,ii));
end
Time_G=toc;

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(solution_G(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_Greedy = counter / length(imgLabelsTest);

counter_Greedy=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_Greedy(ii)=sum(solution_G(ii,:)==imgLabelsTest(ii,:));
end
result_Greedy=zeros(NF,1);
for ii=1:(NF+1)
    result_Greedy(ii)=sum(counter_Greedy==ii-1);
end

precision_Greedy=(0:NF)*result_Greedy/(NF*NUMTEST);

value_Greedy=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_Greedy(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_G(ii,:),opt);
end

end

%% test randomized as comparison
if RANDOM

tic;
solution_R=zeros(size(imgLabelsTest));
parfor ii=1:NUMTEST
    solution_R(ii,:)=Randomized(imgDataTest(:,:,:,ii),solution_G(ii,:));
end
Time_R=toc;

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(solution_R(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_Random = counter / length(imgLabelsTest);

counter_Random=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_Random(ii)=sum(solution_R(ii,:)==imgLabelsTest(ii,:));
end
result_Random=zeros(NF,1);
for ii=1:(NF+1)
    result_Random(ii)=sum(counter_Random==ii-1);
end

precision_Random=(0:NF)*result_Random/(NF*NUMTEST);

value_Random=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_Random(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_R(ii,:),opt);
end

end

filenm=[datestr(now,'dd_mm_yyyy_HH_MM'),'_flow',num2str(NF)];
save([filenm,'.mat']);