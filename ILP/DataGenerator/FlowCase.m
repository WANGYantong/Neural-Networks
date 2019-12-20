function FlowCase(NF)

%% load scenario
% ID of mobile users
flow=1:NF;

NUMINDEX=1024+256;

[para,data] = Scenario(flow);

% NF=length(flow);
NA=length(para.AccessRouter);
NE=length(para.EdgeCloud);
NL=length(para.graph.Edges.EndNodes);

IMAGE=4; % 0 for Constants+Variables
                 % 1 for Variables; 2&3 for Centralized Variables;
                 % 4 for Value normalization
image_layout=ImageEncoding(para.N,NF,NE,NA,NL,IMAGE);
save(['../DataStore/flow',num2str(flow(end)),'/layout.mat'],'image_layout');
imgData=zeros([image_layout.size,1,NUMINDEX]); % dense image size
imgLabels=zeros(NUMINDEX,NF);

result=cell(1,NUMINDEX);

parfor index=1:NUMINDEX %NUMINDEX
    % Generating Training Data
    imgData(:,:,:,index)=DataGenerator(data{index},para,image_layout);
%     imshow(1-imgData(:,:,:,index),'Border','tight','initialMagnification','fit');
    
    % ILP solver
    result{index}=MILP(para,data{index});
    
    % Related Label
    imgLabels(index,:)=result{index}.allocations;
    
    % flag indicates generating process
    fprintf('\nNF:%d; current time: %s; finished iteration: %d',NF,datestr(now, 'mmm.dd,yyyy HH:MM:SS'),index);
    
end

save(['../DataStore/flow',num2str(flow(end)),'/imgData_' num2str(IMAGE) '.mat'],'imgData');
save(['../DataStore/flow',num2str(flow(end)),'/imgLabels_' num2str(IMAGE) '.mat'],'imgLabels');
save(['../DataStore/flow',num2str(flow(end)),'/solutions.mat'],'result');
save(['../DataStore/flow',num2str(flow(end)),'/data.mat'],'data');

time_collection=0;
for ii=1:NUMINDEX
    time_collection=time_collection+result{ii}.time;
end

mean_time=time_collection/NUMINDEX;
save(['time',num2str(flow(end)),'.mat'],'mean_time');

end

