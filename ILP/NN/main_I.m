%% 
clear
clc

addpath(genpath(pwd));

%% load dataset
flow=1:10;
NF=length(flow);

layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
img=load(imgfile);
lab=load(labfile);

training_size=1024;
imgDataTrain=img.imgData(:,:,:,1:training_size);
inputSize=size(imgDataTrain);
imgLabelsTrain=categorical(lab.imgLabels(1:training_size,:));

imgDataTest=img.imgData(:,:,:,7001:7100);
imgLabelsTest=categorical(lab.imgLabels(7001:7100,:));
NUMTEST=size(imgLabelsTest,1);

%% training CNN
layers=[
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

batch_size=64;
epoch_size=10;
learning_rate=1e-3;
options = trainingOptions( 'adam',...
    'ExecutionEnvironment','auto',...
    'MiniBatchSize', batch_size,...
    'MaxEpochs', epoch_size, ...
    'InitialLearnRate',learning_rate,...
    'Verbose',true);
%     'Plots','training-progress');

net = cell(NF,1);
trainInfo = cell(NF,1);
training_clock=tic;
for ii=1:NF
    [net{ii},trainInfo{ii}] = trainNetwork(imgDataTrain, imgLabelsTrain(:,ii), layers, options);
end
training_time=toc(training_clock);
training_time=training_time/NF;

%% testing CNN
predLabelsTestMedium=cell(NF,1);
score=cell(NF,1);
testing_clock=tic;
for ii=1:NF
    [predLabelsTestMedium{ii}, score{ii}] = net{ii}.classify(imgDataTest);
end
predLabelsTest=[predLabelsTestMedium{1:NF}];
% predLabelsTest(isundefined(predLabelsTest))=categorical(1);
scoreTest=[score{1:NF}];
% NE=length(layout.image_layout.space.y);
% scoreTest(isnan(scoreTest))=1/NE;
testing_time=toc(testing_clock);
testing_time=testing_time/NUMTEST;

testing_accuracy=zeros(6,1);
opt.NF=NF;
opt.NT=NUMTEST;
for jj=1:length(testing_accuracy)
    opt.mode=jj;
    testing_accuracy(jj)=ErrorCalc(imgLabelsTest, score, opt);
end

%% Hill Climbing Algorithm
alloc_HC=cell(NUMTEST,1);
value_HC=zeros(NUMTEST,1);

climbing_clock=tic;
for ii=1:NUMTEST
    [buff_HC,value_HC(ii)]=HillClimbing(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
    alloc_HC{ii}=buff_HC';
end
climbing_time=toc(climbing_clock);
climbing_time=climbing_time/NUMTEST;

alloc_hill=[alloc_HC{1:NUMTEST}]';

counter=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter(ii)=sum(alloc_hill(ii,:)==categorical(imgLabelsTest(ii,:)));
end
counter_NF=zeros(NF,1);
for ii=1:(NF+1)
    counter_NF(ii)=sum(counter==ii-1);
end

hill_accuracy=(0:NF)*counter_NF/(NF*NUMTEST);
hill_time=testing_time+climbing_time;
%% sub MILP resolving
result_sM=cell(NUMTEST,1);
alloc_sM=cell(NUMTEST,1);
value_sM=zeros(NUMTEST,1);

subM_clock=tic;
for ii=1:NUMTEST
    result_sM{ii}=subMILP(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
    alloc_sM{ii}=result_sM{ii}.allocations;
    value_sM(ii)=result_sM{ii}.fval;
end
subM_time=toc(subM_clock);
subM_time=subM_time/NUMTEST;

alloc_sub=[alloc_sM{1:NUMTEST}]';

counter=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter(ii)=sum(alloc_sub(ii,:)==categorical(imgLabelsTest(ii,:)));
end
counter_NF=zeros(NF,1);
for ii=1:(NF+1)
    counter_NF(ii)=sum(counter==ii-1);
end

sub_accuracy=(0:NF)*counter_NF/(NF*NUMTEST);
sub_time=testing_time+subM_time;