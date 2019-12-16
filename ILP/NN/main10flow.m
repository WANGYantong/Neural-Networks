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

testing_range=9001:9100;
imgDataTest=img.imgData(:,:,:,testing_range);
imgLabelsTest=categorical(lab.imgLabels(testing_range,:));
NUMTEST=length(testing_range);

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

%% CNN+Hill Climbing
% value_hill, alloc_hill, accuracy_hill, time_hill
alloc_HC=cell(NUMTEST,1);
value_hill=zeros(NUMTEST,1);

climbing_clock=tic;
for ii=1:NUMTEST
    [buff_HC,value_hill(ii)]=HillClimbing(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
    alloc_HC{ii}=buff_HC';
end
climbing_time=toc(climbing_clock);
climbing_time=climbing_time/NUMTEST;
disp('CNN+Hill Climbing finished');

alloc_hill=[alloc_HC{1:NUMTEST}]';

counter=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter(ii)=sum(alloc_hill(ii,:)==categorical(imgLabelsTest(ii,:)));
end
counter_NF=zeros(NF,1);
for ii=1:(NF+1)
    counter_NF(ii)=sum(counter==ii-1);
end

accuracy_hill=(0:NF)*counter_NF/(NF*NUMTEST);
time_hill=testing_time+climbing_time;

%% CNN+sub MILP resolving
% value_sub, alloc_sub, accuracy_sub, time_sub
result_sM=cell(NUMTEST,1);
alloc_sM=cell(NUMTEST,1);
value_sub=zeros(NUMTEST,1);

subM_clock=tic;
for ii=1:NUMTEST
    result_sM{ii}=subMILP(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:));
    alloc_sM{ii}=result_sM{ii}.allocations;
    value_sub(ii)=result_sM{ii}.fval;
end
subM_time=toc(subM_clock);
subM_time=subM_time/NUMTEST;
disp('CNN+sub MILP finished');

alloc_sub=[alloc_sM{1:NUMTEST}]';

counter=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter(ii)=sum(alloc_sub(ii,:)==categorical(imgLabelsTest(ii,:)));
end
counter_NF=zeros(NF,1);
for ii=1:(NF+1)
    counter_NF(ii)=sum(counter==ii-1);
end

accuracy_sub=(0:NF)*counter_NF/(NF*NUMTEST);
time_sub=testing_time+subM_time;

%% Greedy Algorithm
% value_greedy, alloc_greedy, accuracy_greedy, time_greedy
result_G=cell(NUMTEST,1);
alloc_G=cell(NUMTEST,1);
value_greedy=zeros(NUMTEST,1);

G_clock=tic;
for ii=1:NUMTEST
    result_G{ii}=Greedy(imgDataTest(:,:,:,ii));
    alloc_G{ii}=result_G{ii}.allocations;
    value_greedy(ii)=result_G{ii}.value;
end
G_time=toc(G_clock);
time_greedy=G_time/NUMTEST;
disp('Greedy finished');

alloc_greedy=[alloc_G{1:NUMTEST}]';

counter=zeros(NUMTEST,1);
for ii=1:NUMTEST
    counter(ii)=sum(alloc_greedy(ii,:)==categorical(imgLabelsTest(ii,:)));
end
counter_NF=zeros(NF,1);
for ii=1:(NF+1)
    counter_NF(ii)=sum(counter==ii-1);
end

accuracy_greedy=(0:NF)*counter_NF/(NF*NUMTEST);

%% Benchmark
% value_bench, alloc_bench, time_bench
sol=load(['../DataStore/flow',num2str(flow(end)),'/solutions.mat']);
value_bench=zeros(NUMTEST,1);
alloc_bench=imgLabelsTest;
time_bench=zeros(NUMTEST,1);

for ii=1:NUMTEST
    result_bench=sol.result{testing_range(ii)};
    value_bench(ii)=result_bench.fval;
    time_bench(ii)=result_bench.time;
end
disp('Benchmark loaded');

time_bench=mean(time_bench);