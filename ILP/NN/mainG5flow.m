%% 
clear
clc

MONTECARLO=100;

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
flow=1:10;
NF=length(flow);

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

for index_monte=1:MONTECARLO
    
load(['Net\net',num2str(index_monte),'.mat']);

NI=NF/size(net,1); % number of iterations, must be an integer
% throw out error if not ingeter
if floor(NI) ~= NI
    error('NI is not an integer, not supported yet...');
end

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

predLabelsTestMedium = cell(NF,1);
score = cell(NF,1);
imgDataCopy=imgDataTest;
testing_clock=tic;
for ii=1:NI
    for jj=1:size(net,1)
        index=size(net,1)*(ii-1)+jj; % index of flows
        [predLabelsTestMedium{index}, score{index}] = net{jj}.classify(imgDataCopy(5*(ii-1)+1:5*(ii-1)+size(net,1),:,:,:));
    end
    % update image
    imgDataCopy=imgUpdate(imgDataTest,predLabelsTestMedium,layout,Net);
end
predLabelsTest=[predLabelsTestMedium{1:NF}];
for ii=1:numel(predLabelsTest)
    if isundefined(predLabelsTest(ii))
        predLabelsTest(ii)=categorical(randi(NE));
    end
end
scoreTest=[score{1:NF}];
scoreTest(isnan(scoreTest))=1/NE;
testing_time=toc(testing_clock);
testing_time=testing_time/NUMTEST;

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
predLabelsTestMedium_HCLS = cell(NF,1);
score_HCLS = cell(NF,1);
imgDataCopy_HCLS=imgDataTest;
clock_HCLS=tic;
for ii=1:NI
    image_range=5*(ii-1)+1:5*(ii-1)+size(net,1);
    % initial arrangement
    for jj=1:size(net,1)
        index=size(net,1)*(ii-1)+jj; % index of flows
        [predLabelsTestMedium_HCLS{index}, score_HCLS{index}] = net{jj}.classify(imgDataCopy_HCLS(image_range,:,:,:));
    end
    % update arrangement
    predLabelsTest_HCLS=[predLabelsTestMedium_HCLS{image_range}];
    for kk=1:numel(predLabelsTest_HCLS)
        if isundefined(predLabelsTest_HCLS(kk))
            predLabelsTest_HCLS(kk)=categorical(randi(NE));
        end
    end
    scoreTest_HCLS=[score_HCLS{image_range}];
    scoreTest_HCLS(isnan(scoreTest_HCLS))=1/NE;
    
    predLabelsTestMedium_HCLS(image_range)=...
        HillClimbing_New(imgDataCopy_HCLS(image_range,:,:,:),predLabelsTest_HCLS,scoreTest_HCLS,Net);
    % update image
    imgDataCopy_HCLS=imgUpdate(imgDataTest,predLabelsTestMedium_HCLS,layout,Net);
end
time_HCLS=toc(clock_HCLS);

% computation time & mean TC & feasible ratio
TC_HCLS=zeros(NUMTEST,1);
feasible_HCLS=TC_HCLS;

alloc_HCLS=[predLabelsTestMedium_HCLS{1:NF}];

for ii=1:NUMTEST
    [TC_HCLS(ii),feasible_HCLS(ii)]=TCcalculator(imgDataTest(:,:,:,ii),alloc_HCLS(ii,:));
end

meanTime_HCLS=time_HCLS/NUMTEST;
meanTC_HCLS=mean(TC_HCLS);
meanFeasible_HCLS=mean(feasible_HCLS);

% max TC diff
TCdiff_HCLS=TC_HCLS-TC_MILP;

% accuracy + precision + recall + F1-score
[TP_HCLS,FP_HCLS,TN_HCLS,FN_HCLS]=...
        ConfusionMatrix(imgLabelsTest,alloc_HCLS);

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
predLabelsTestMedium_MILP = cell(NF,1);
score_MILP = cell(NF,1);
imgDataCopy_MILP=imgDataTest;

TC_CNNMILP=zeros(NUMTEST,1);
saveDV_CNNMILP=zeros(NUMTEST,1);

clock_MILP=tic;
for ii=1:NI
    image_range=5*(ii-1)+1:5*(ii-1)+size(net,1);
    % initial arrangement
    for jj=1:size(net,1)
        index=size(net,1)*(ii-1)+jj; % index of flows
        [predLabelsTestMedium_MILP{index}, score_MILP{index}] = net{jj}.classify(imgDataCopy_MILP(image_range,:,:,:));
    end
    % update arrangement
    predLabelsTest_MILP=[predLabelsTestMedium_MILP{image_range}];
    for kk=1:numel(predLabelsTest_MILP)
        if isundefined(predLabelsTest_MILP(kk))
            predLabelsTest_MILP(kk)=categorical(randi(NE));
        end
    end
    scoreTest_MILP=[score_MILP{image_range}];
    scoreTest_MILP(isnan(scoreTest_MILP))=1/NE;
    
    [predLabelsTestMedium_MILP(image_range),buffTC_CNNMILP,buffDV_CNNMILP]=...
        subMILP_New(imgDataCopy_MILP(image_range,:,:,:),predLabelsTest_MILP,scoreTest_MILP,Net);
    % update image
    imgDataCopy_MILP=imgUpdate(imgDataTest,predLabelsTestMedium_MILP,layout,Net);
    
    TC_CNNMILP=TC_CNNMILP+buffTC_CNNMILP;
    saveDV_CNNMILP=saveDV_CNNMILP+buffDV_CNNMILP;
end
time_CNNMILP=toc(clock_MILP);

% computation time & mean TC & number of decision variables
meanTime_CNNMILP=time_CNNMILP/NUMTEST;
meanTC_CNNMILP=mean(TC_CNNMILP);
meanNumberDV_CNNMILP=mean(numberDV_MILP-saveDV_CNNMILP);

% max TC diff
TCdiff_CNNMILP=TC_CNNMILP-TC_MILP;

% accuracy + precision + recall + F1-score
predLabels_CNNMILP=[predLabelsTestMedium_MILP{1:NF}];

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
disp(mean(cell2mat(meanTime_Monte)));
disp('TC');
disp(mean(cell2mat(meanTC_Monte)));
disp('ratio');
disp(mean(cell2mat(meanFeasible_Monte)));
disp('diff');
disp(mean(cell2mat(maxDiff_Monte)));
disp('d.v.');
disp(mean(cell2mat(numDV_Monte)));
disp('macro acc');
disp(mean(cell2mat(macroAcc_Monte)));
disp('macro pre');
disp(mean(cell2mat(macroPre_Monte)));
disp('macro rec');
disp(mean(cell2mat(macroRec_Monte)));
disp('macro f1');
disp(mean(cell2mat(macroF1_Monte)));
disp('micro acc');
disp(mean(cell2mat(microAcc_Monte)));
disp('micro pre');
disp(mean(cell2mat(microPre_Monte)));
disp('micro rec');
disp(mean(cell2mat(microRec_Monte)));
disp('micro f1');
disp(mean(cell2mat(microF1_Monte)));

save('plot_data\10flow.mat');