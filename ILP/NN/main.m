%%
clear
clc

addpath(genpath(pwd));

GRC=1;
RGC=1;
RAN=1;
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
        'LearnRateSchedule','piecewise',...
        'LearnRateDropPeriod',10,...
        'LearnRateDropFactor',0.1,...
        'GradientDecayFactor',0.9,...
        'SquaredGradientDecayFactor',0.999,...
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
% throw out error if not ingeter
if floor(NI) ~= NI
    error('NI is not an integer, not supported yet...');
end
    
predLabelsTestMedium = cell(NF,1);
score = cell(NF,1);
imgDataCopy=imgDataTest;
watch_tog=tic;
for ii=1:NI
    for jj=1:size(net,1)
        index=size(net,1)*(ii-1)+jj; % index of flows
        [predLabelsTestMedium{index}, score{index}] = net{jj}.classify(imgDataCopy(5*(ii-1)+1:5*(ii-1)+size(net,1),:,:,:));
    end
    % update image
    imgDataCopy=imgUpdate(imgDataTest,predLabelsTestMedium,layout);
end
predLabelsTest=[predLabelsTestMedium{1:NF}];
predLabelsTest(isundefined(predLabelsTest))=categorical(1);
scoreTest=[score{1:NF}];
NE=length(layout.image_layout.space.y);
scoreTest(isnan(scoreTest))=1/NE;

watch_MILP=tic;
for ii=1:NUMTEST
%     predLabelsTest(ii,:)=combiner_I(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:), 2);
%     predLabelsTest(ii,:)=combiner_II(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
    predLabelsTest(ii,:)=combiner_III(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
end
% predLabelsTest=net.predict(imgDataTest);
running_time_MILP=toc(watch_MILP);

running_time_tog=toc(watch_tog);

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(predLabelsTest(jj,:)==categorical(imgLabelsTest(jj,:)))
%     if all(round(predLabelsTest(jj,:))==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_final = counter / size(imgLabelsTest,1);

accuracy=zeros(1,NF);
for ii=1:NF
    accuracy(ii) = sum(predLabelsTestMedium{ii} ==categorical(imgLabelsTest(:,ii))) / size(imgLabelsTest,1);
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

%% GReedy Caching
if GRC

tic;
solution_GRC=zeros(size(imgLabelsTest));
for ii=1:NUMTEST
    solution_GRC(ii,:)=Greedy(imgDataTest(:,:,:,ii));
end
Time_GRC=toc;

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(solution_GRC(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_GRC = counter / length(imgLabelsTest);

counter_GRC=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_GRC(ii)=sum(solution_GRC(ii,:)==imgLabelsTest(ii,:));
end
result_GRC=zeros(NF,1);
for ii=1:(NF+1)
    result_GRC(ii)=sum(counter_GRC==ii-1);
end

precision_GRC=(0:NF)*result_GRC/(NF*NUMTEST);

value_GRC=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_GRC(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_GRC(ii,:),opt);
end

end

%% test Randomized Greedy Caching as comparison
if RGC
    
solution_RGC=zeros(size(imgLabelsTest));

% solution_RGC_buff=solution_RGC;
tic;
parfor ii=1:NUMTEST
    solution_RGC(ii,:)=Randomized(imgDataTest(:,:,:,ii),solution_GRC(ii,:));
%     solution_RGC(ii,:)=Randomized(imgDataTest(:,:,:,ii),solution_RGC_buff(ii,:));
end
Time_RGC=toc;

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(solution_RGC(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_RGC = counter / length(imgLabelsTest);

counter_RGC=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_RGC(ii)=sum(solution_RGC(ii,:)==imgLabelsTest(ii,:));
end
result_RGC=zeros(NF,1);
for ii=1:(NF+1)
    result_RGC(ii)=sum(counter_RGC==ii-1);
end

precision_RGC=(0:NF)*result_RGC/(NF*NUMTEST);

value_RGC=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_RGC(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_RGC(ii,:),opt);
end

end

%% Randomized assignment
if RAN
    
tic;
solution_RAN=zeros(size(imgLabelsTest));
for ii=1:NUMTEST
    solution_RAN(ii,:)=randi([1,NE],1,NF);
end
Time_RAN=toc;

counter=0;
for jj=1:size(imgLabelsTest,1)
    if all(solution_RAN(jj,:)==imgLabelsTest(jj,:))
        counter=counter+1;
    end
end
accuracy_RAN = counter / length(imgLabelsTest);

counter_RAN=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter_RAN(ii)=sum(solution_RAN(ii,:)==imgLabelsTest(ii,:));
end
result_RAN=zeros(NF,1);
for ii=1:(NF+1)
    result_RAN(ii)=sum(counter_RAN==ii-1);
end

precision_RAN=(0:NF)*result_RAN/(NF*NUMTEST);

value_RAN=zeros(NUMTEST,1);
opt.mode=0;
for ii=1:NUMTEST
    value_RAN(ii)=valueCalculator(imgDataTest(:,:,:,ii),solution_RAN(ii,:),opt);
end
    
end

% filenm=[datestr(now,'dd_mm_yyyy_HH_MM'),'_flow',num2str(NF)];
% save([filenm,'.mat']);