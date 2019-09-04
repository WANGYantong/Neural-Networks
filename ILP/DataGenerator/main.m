%% preprocess
clear
clc

%% topology
[G,EdgeCloud,NormalRouter,AccessRouter,vertice_names]=SetNetTopo();
N=length(vertice_names);
for v=1:N
    eval([vertice_names{v},'=',num2str(v),';']);
end

%% setting parameter
% ID of mobile users
global flow;
flow=1:5;
% caching cost per EC
alpha=0.5;
% transmission cost per hop
beta=1;
% cache miss penalty
gamma=1;

NF=length(flow);
NA=length(AccessRouter);
NE=length(EdgeCloud);
NL=length(G.Edges.EndNodes);

NUMINDEX=5000;

IMAGE=4; % 0 for Constants+Variables
                 % 1 for Variables; 2&3 for Centralized Variables;
                 % 4 for Value normalization
image_layout=ImageEncoding(N,NF,NE,NA,NL,IMAGE);
save(['../DataStore/flow',num2str(flow(end)),'/layout.mat'],'image_layout');
imgData=zeros([image_layout.size,1,NUMINDEX]); % dense image size
imgLabels=zeros(NUMINDEX,NF);

% # of hops from AR to EC
hopcounter=zeros(length(AccessRouter),length(EdgeCloud));
path=cell(length(AccessRouter),length(EdgeCloud));
for ii=1:length(AccessRouter)
    for jj=1:length(EdgeCloud)
        [path{ii,jj},hopcounter(ii,jj)]=shortestpath(G,AccessRouter(ii),EdgeCloud(jj));
    end
end
% # of hops from AR to DataCenter
hoptotal=15;
% total space in EC
spaceT=ones(size(EdgeCloud))*500;
% total bandwidth in link
bandwidthT=ones(size(G.Edges.Weight))*1000;
% relationship between node and link
B=GetPathLinkRel(G,"undirected",path,length(AccessRouter),length(EdgeCloud));
% surfficiently large number
M=1000;

save(['../DataStore/flow',num2str(flow(end)),'/network.mat'],'alpha','beta','gamma','hopcounter','hoptotal','B','G',...
    'EdgeCloud','AccessRouter');

para.graph=G;
para.EdgeCloud=EdgeCloud;
para.AccessRouter=AccessRouter;
para.NormalRouter=[GW,NormalRouter];

result=cell(NUMINDEX,1);
data=cell(NUMINDEX,1);

for index=1:NUMINDEX
    
    rng(index);
    
    % moving probability
    [probability,start_point]=SetMovProb(length(flow),length(AccessRouter));
    % space requirement of flow
    spaceK=randi([0,5],size(flow))*10+45;
    % available space in EC
    spaceR=[randi([0,4],size(EdgeCloud(1:2)))*100+600,randi([0,2],size(EdgeCloud(3:end)))*100+200];
    % bandwidth requirement of flow
    bandwidthK=randi([0,2],size(flow))*5+5;
    % available bandwidth in link
    bandwidthR=randi([0,2],size(G.Edges.Weight))*10+80;
%     bandwidthR=ones(size(G.Edges.Weight))*100;
    
    % packing parameters
    data{index}.flow=flow;
    data{index}.alpha=alpha;
    data{index}.beta=beta;
    data{index}.gamma=gamma;
    data{index}.probability=probability;
    data{index}.startPoint=start_point;
    data{index}.hopcounter=hopcounter;
    data{index}.path=path;
    data{index}.hoptotal=hoptotal;
    data{index}.spaceK=spaceK;
    data{index}.spaceR=spaceR;
    data{index}.spaceT=spaceT;
    data{index}.bandwidthK=bandwidthK;
    data{index}.bandwidthR=bandwidthR;
    data{index}.bandwidthT=bandwidthT;
    data{index}.B=B;
    data{index}.M=M;
        
end

parfor index=1:NUMINDEX
    % Generating Training Data
    imgData(:,:,:,index)=DataGenerator(data{index},para,image_layout);
%     imshow(1-imgData(:,:,:,index),'Border','tight','initialMagnification','fit');
    
    % ILP solver
    result{index}=MILP(para,data{index});
    
    % Related Label
    imgLabels(index,:)=result{index}.allocations;
end

save(['../DataStore/flow',num2str(flow(end)),'/imgData_' num2str(IMAGE) '.mat'],'imgData');
save(['../DataStore/flow',num2str(flow(end)),'/imgLabels_' num2str(IMAGE) '.mat'],'imgLabels');
save(['../DataStore/flow',num2str(flow(end)),'/solutions.mat'],'result');
save(['../DataStore/flow',num2str(flow(end)),'/data.mat'],'data');