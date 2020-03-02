%% 
clear
clc

MONTECARLO=20;

meanTime_Monte=cell(MONTECARLO,1);
meanTC_Monte=meanTime_Monte;
meanFeasible_Monte=meanTime_Monte;
maxDiff_Monte=meanTime_Monte;
numDV_Monte=meanTime_Monte;
macroAcc_Monte=meanTime_Monte;
macroPre_Monte=meanTime_Monte;
macroRec_Monte=meanTime_Monte;
macroF1_Monte=meanTime_Monte;
microAcc_Monte=meanTime_Monte;
microPre_Monte=meanTime_Monte;
microRec_Monte=meanTime_Monte;
microF1_Monte=meanTime_Monte;

addpath(genpath(pwd));

%% load dataset
flow=1:5;
NF=length(flow);

layout=load(['../DataStore/flow',num2str(NF),'/layout.mat']);
imgfile=['../DataStore/flow',num2str(NF),'/imgData_',num2str(layout.image_layout.opts),'.mat'];
labfile=['../DataStore/flow',num2str(NF),'/imgLabels_',num2str(layout.image_layout.opts),'.mat'];
img=load(imgfile);
lab=load(labfile);

training_size=1024;

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

for index_monte=1:MONTECARLO
%% training CNN
layers=[
    imageInputLayer(inputSize(1:3))
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
%     convolution2dLayer(3,16,'Padding','same')
%     batchNormalizationLayer
%     reluLayer

    fullyConnectedLayer(numel(unique(imgLabelsTrain)))
    softmaxLayer
    classificationLayer];

batch_size=64;
epoch_size=30;
learning_rate=1e-3;

options = trainingOptions(  'sgdm',...
        'ExecutionEnvironment','auto',...
        'MiniBatchSize', batch_size,...
        'MaxEpochs', epoch_size, ...
        'Shuffle','every-epoch',...
        'InitialLearnRate',learning_rate,...
        'L2Regularization',0.0005,...
        'Verbose',false);
%         'Plots','training-progress');

net = cell(NF,1);
training_clock=tic;
for ii=1:NF
    net{ii} = trainNetwork(imgDataTrain, imgLabelsTrain(:,ii), layers, options);
end
training_time=toc(training_clock);
training_time=training_time/NF;

save(['Net\net',num2str(index_monte),'.mat'],'net');
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

%% Benchmark
sol=load(['../DataStore/flow',num2str(flow(end)),'/solutions.mat']);

% computation time & mean_TC
time_MILP=zeros(NUMTEST,1);
TC_MILP=zeros(NUMTEST,1);
for ii=1:NUMTEST
    result_bench=sol.result{testing_range(ii)};
    TC_MILP(ii)=result_bench.fval;
    time_MILP(ii)=result_bench.time;
end

meanTime_MILP=mean(time_MILP);
meanTC_MILP=mean(TC_MILP);

% # of decision variables
Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[NL,NA,NE]=size(Net.B);

numberDV_MILP=NF*NE+NF*NL+NF*NA*NE+NE+NF*NE;
%% pure-CNN

% computation time
meanTime_pure=testing_time;

% mean TC & feasible ratio
TC_pure=zeros(NUMTEST,1);
feasible_pure=TC_pure;
for ii=1:NUMTEST
    [TC_pure(ii),feasible_pure(ii)]=TCcalculator(imgDataTest(:,:,:,ii),predLabelsTest(ii,:));
end

meanTC_pure=mean(TC_pure);
meanFeasible_pure=mean(feasible_pure);

% max TC diff
TCdiff_pure=TC_pure-TC_MILP;

% accuracy + precision + recall + F1-score
[TP_pure,FP_pure,TN_pure,FN_pure]=...
        ConfusionMatrix(imgLabelsTest,predLabelsTest);
    
[MacroAcc_pure,MacroPre_pure,MacroRec_pure,MacroF1_pure]=...
    MacroAveraging(TP_pure,FP_pure,TN_pure,FN_pure);

[MicroAcc_pure,MicroPre_pure,MicroRec_pure,MicroF1_pure]=...
    MicroAveraging(TP_pure,FP_pure,TN_pure,FN_pure);
% testing_accuracy=zeros(6,1);
% opt.NF=NF;
% opt.NT=NUMTEST;
% for jj=1:length(testing_accuracy)
%     opt.mode=jj;
%     testing_accuracy(jj)=ErrorCalc(imgLabelsTest, score, opt);
% end

%% CNN+HCLS
alloc_HCLS=cell(NUMTEST,1);
TC_HCLS=zeros(NUMTEST,1);
feasible_HCLS=TC_HCLS;

% computation time & mean TC & feasible ratio
clock_HCLS=tic;
for ii=1:NUMTEST
    [buff_HCLS,TC_HCLS(ii),feasible_HCLS(ii)]=...
        HillClimbing(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:),Net);
    alloc_HCLS{ii}=buff_HCLS';
end
time_HCLS=toc(clock_HCLS);

meanTime_HCLS=time_HCLS/NUMTEST+testing_time;
meanTC_HCLS=mean(TC_HCLS);
meanFeasible_HCLS=mean(feasible_HCLS);

% max TC diff
TCdiff_HCLS=TC_HCLS-TC_MILP;

% accuracy + precision + recall + F1-score
predLabels_HCLS=[alloc_HCLS{1:NUMTEST}]';

[TP_HCLS,FP_HCLS,TN_HCLS,FN_HCLS]=...
        ConfusionMatrix(imgLabelsTest,predLabels_HCLS);

[MacroAcc_HCLS,MacroPre_HCLS,MacroRec_HCLS,MacroF1_HCLS]=...
    MacroAveraging(TP_HCLS,FP_HCLS,TN_HCLS,FN_HCLS);

[MicroAcc_HCLS,MicroPre_HCLS,MicroRec_HCLS,MicroF1_HCLS]=...
    MicroAveraging(TP_HCLS,FP_HCLS,TN_HCLS,FN_HCLS);


% counter=zeros(NUMTEST,1);
% for ii=1:NUMTEST
%     counter(ii)=sum(alloc_hill(ii,:)==categorical(imgLabelsTest(ii,:)));
% end
% counter_NF=zeros(NF,1);
% for ii=1:(NF+1)
%     counter_NF(ii)=sum(counter==ii-1);
% end
% 
% accuracy_hill=(0:NF)*counter_NF/(NF*NUMTEST);
% time_hill=testing_time+climbing_time;

%% CNN-MILP 

result_CNNMILP=cell(NUMTEST,1);
alloc_CNNMILP=cell(NUMTEST,1);
TC_CNNMILP=zeros(NUMTEST,1);
numberDV_CNNMILP=zeros(NUMTEST,1);
time_CNNMILP=zeros(NUMTEST,1);

% computation time & mean TC & number of decision variables
for ii=1:NUMTEST
    result_CNNMILP{ii}=subMILP(imgDataTest(:,:,:,ii), predLabelsTest(ii,:), scoreTest(ii,:),Net);
    alloc_CNNMILP{ii}=result_CNNMILP{ii}.allocations;
    TC_CNNMILP(ii)=result_CNNMILP{ii}.fval;
    numberDV_CNNMILP(ii)=numberDV_MILP-result_CNNMILP{ii}.num_var;
    time_CNNMILP(ii)=result_CNNMILP{ii}.time;
end

meanTime_CNNMILP=mean(time_CNNMILP)+testing_time;
meanTC_CNNMILP=mean(TC_CNNMILP);
meanNumberDV_CNNMILP=mean(numberDV_CNNMILP);

% max TC diff
TCdiff_CNNMILP=TC_CNNMILP-TC_MILP;

% accuracy + precision + recall + F1-score
predLabels_CNNMILP=[alloc_CNNMILP{1:NUMTEST}]';

[TP_CNNMILP,FP_CNNMILP,TN_CNNMILP,FN_CNNMILP]=...
        ConfusionMatrix(imgLabelsTest,predLabels_CNNMILP);

[MacroAcc_CNNMILP,MacroPre_CNNMILP,MacroRec_CNNMILP,MacroF1_CNNMILP]=...
    MacroAveraging(TP_CNNMILP,FP_CNNMILP,TN_CNNMILP,FN_CNNMILP);

[MicroAcc_CNNMILP,MicroPre_CNNMILP,MicroRec_CNNMILP,MicroF1_CNNMILP]=...
    MicroAveraging(TP_CNNMILP,FP_CNNMILP,TN_CNNMILP,FN_CNNMILP);

% counter=zeros(NUMTEST,1);
% for ii=1:NUMTEST
%     counter(ii)=sum(alloc_sub(ii,:)==categorical(imgLabelsTest(ii,:)));
% end
% counter_NF=zeros(NF,1);
% for ii=1:(NF+1)
%     counter_NF(ii)=sum(counter==ii-1);
% end
% 
% accuracy_sub=(0:NF)*counter_NF/(NF*NUMTEST);
% time_sub=testing_time+subM_time;

%% Greedy Algorithm

% computation time & mean TC & feasible ratio
result_Greedy=cell(NUMTEST,1);
alloc_Greedy=cell(NUMTEST,1);
TC_Greedy=zeros(NUMTEST,1);
feasible_Greedy=zeros(NUMTEST,1);

clock_Greedy=tic;
for ii=1:NUMTEST
    result_Greedy{ii}=Greedy(imgDataTest(:,:,:,ii),Net);
    alloc_Greedy{ii}=result_Greedy{ii}.allocations;
    TC_Greedy(ii)=result_Greedy{ii}.value;
    feasible_Greedy(ii)=result_Greedy{ii}.ratio;
end
time_Greedy=toc(clock_Greedy);

meanTime_Greedy=time_Greedy/NUMTEST;
meanTC_Greedy=mean(TC_Greedy);
meanFeasible_Greedy=mean(feasible_Greedy);

% max TC diff
TCdiff_Greedy=TC_Greedy-TC_MILP;

% accuracy + precision + recall + F1-score
predLabels_Greedy=[alloc_Greedy{1:NUMTEST}]';

[TP_Greedy,FP_Greedy,TN_Greedy,FN_Greedy]=...
        ConfusionMatrix(imgLabelsTest,predLabels_Greedy);

[MacroAcc_Greedy,MacroPre_Greedy,MacroRec_Greedy,MacroF1_Greedy]=...
    MacroAveraging(TP_Greedy,FP_Greedy,TN_Greedy,FN_Greedy);

[MicroAcc_Greedy,MicroPre_Greedy,MicroRec_Greedy,MicroF1_Greedy]=...
    MicroAveraging(TP_Greedy,FP_Greedy,TN_Greedy,FN_Greedy);

% counter=zeros(NUMTEST,1);
% for ii=1:NUMTEST
%     counter(ii)=sum(alloc_greedy(ii,:)==categorical(imgLabelsTest(ii,:)));
% end
% counter_NF=zeros(NF,1);
% for ii=1:(NF+1)
%     counter_NF(ii)=sum(counter==ii-1);
% end
% 
% accuracy_greedy=(0:NF)*counter_NF/(NF*NUMTEST);
meanTime_Monte{index_monte}=[meanTime_MILP,meanTime_pure,meanTime_CNNMILP,meanTime_HCLS,meanTime_Greedy];
meanTC_Monte{index_monte}=[meanTC_MILP,meanTC_pure,meanTC_CNNMILP,meanTC_HCLS,meanTC_Greedy];
meanFeasible_Monte{index_monte}=[meanFeasible_pure,meanFeasible_HCLS,meanFeasible_Greedy];
maxDiff_Monte{index_monte}=[max(TCdiff_pure),max(TCdiff_CNNMILP),max(TCdiff_HCLS),max(TCdiff_Greedy)];
numDV_Monte{index_monte}=[numberDV_MILP,meanNumberDV_CNNMILP];
macroAcc_Monte{index_monte}=[MacroAcc_pure,MacroAcc_CNNMILP,MacroAcc_HCLS,MacroAcc_Greedy];
macroPre_Monte{index_monte}=[MacroPre_pure,MacroPre_CNNMILP,MacroPre_HCLS,MacroPre_Greedy];
macroRec_Monte{index_monte}=[MacroRec_pure,MacroRec_CNNMILP,MacroRec_HCLS,MacroRec_Greedy];
macroF1_Monte{index_monte}=[MacroF1_pure,MacroF1_CNNMILP,MacroF1_HCLS,MacroF1_Greedy];
microAcc_Monte{index_monte}=[MicroAcc_pure,MicroAcc_CNNMILP,MicroAcc_HCLS,MicroAcc_Greedy];
microPre_Monte{index_monte}=[MicroPre_pure,MicroPre_CNNMILP,MicroPre_HCLS,MicroPre_Greedy];
microRec_Monte{index_monte}=[MicroRec_pure,MicroRec_CNNMILP,MicroRec_HCLS,MicroRec_Greedy];
microF1_Monte{index_monte}=[MicroF1_pure,MicroF1_CNNMILP,MicroF1_HCLS,MicroF1_Greedy];

fprintf('\n number %d simulation \n',index_monte);

end

disp('time');
mean_meanTime_Monte=mean(cell2mat(meanTime_Monte));
disp(mean_meanTime_Monte);
disp('TC');
mean_meanTC_Monte=mean(cell2mat(meanTC_Monte));
disp(mean_meanTC_Monte);
disp('ratio');
mean_meanFeasible_Monte=mean(cell2mat(meanFeasible_Monte));
disp(mean_meanFeasible_Monte);
disp('diff');
mean_maxDiff_Monte=mean(cell2mat(maxDiff_Monte));
disp(mean_maxDiff_Monte);
disp('d.v.');
mean_numDV_Monte=mean(cell2mat(numDV_Monte));
disp(mean_numDV_Monte);
disp('macro acc');
mean_macroAcc_Monte=mean(cell2mat(macroAcc_Monte));
disp(mean_macroAcc_Monte);
disp('macro pre');
mean_macroPre_Monte=mean(cell2mat(macroPre_Monte));
disp(mean_macroPre_Monte);
disp('macro rec');
mean_macroRec_Monte=mean(cell2mat(macroRec_Monte));
disp(mean_macroRec_Monte);
disp('macro f1');
mean_macroF1_Monte=mean(cell2mat(macroF1_Monte));
disp(mean_macroF1_Monte);
disp('micro acc');
mean_microAcc_Monte=mean(cell2mat(microAcc_Monte));
disp(mean_microAcc_Monte);
disp('micro pre');
mean_microPre_Monte=mean(cell2mat(microPre_Monte));
disp(mean_microPre_Monte);
disp('micro rec');
mean_microRec_Monte=mean(cell2mat(microRec_Monte));
disp(mean_microRec_Monte);
disp('micro f1');
mean_microF1_Monte=mean(cell2mat(microF1_Monte));
disp(mean_microF1_Monte);

save('plot_data\5flow.mat');