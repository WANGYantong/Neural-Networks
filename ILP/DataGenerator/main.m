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
flow=1:10;
% caching cost per EC
alpha=0.8;
% transmission cost per hop
beta=0.2;

NF=length(flow);
NA=length(AccessRouter);
NE=length(EdgeCloud);
NL=length(G.Edges.EndNodes);

NUMINDEX=10000;

IMAGE=4; % 0 for Constants+Variables
                 % 1 for Variables; 2&3 for Centralized Variables;
                 % 4 for Value normalization
image_layout=ImageEncoding(N,NF,NE,NA,NL,IMAGE);
save('../DataStore/layout.mat','image_layout');
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
spaceT=ones(size(EdgeCloud))*50;
% total bandwidth in link
bandwidthT=ones(size(G.Edges.Weight))*100;
% relationship between node and link
B=GetPathLinkRel(G,"undirected",path,length(AccessRouter),length(EdgeCloud));
% surfficiently large number
M=1000;

save('../DataStore/network.mat','alpha','beta','hopcounter','hoptotal','B','G',...
    'EdgeCloud','AccessRouter');

for index=1:NUMINDEX
    
    rng(index);
    
    % moving probability
    [probability,start_point]=SetMovProb(length(flow),length(AccessRouter));
    % space requirement of flow
    spaceK=randi(10,size(flow));
    % available space in EC
    spaceR=randi(40,size(EdgeCloud))+10;
    % bandwidth requirement of flow
    bandwidthK=randi(10,size(flow));
    % available bandwidth in link
    bandwidthR=randi(50,size(G.Edges.Weight))+50;
    
    %% packing parameters
    para.graph=G;
    para.EdgeCloud=EdgeCloud;
    para.AccessRouter=AccessRouter;
    para.NormalRouter=[GW,NormalRouter];
    
    data.flow=flow;
    data.alpha=alpha;
    data.beta=beta;
    data.probability=probability;
    data.startPoint=start_point;
    data.hopcounter=hopcounter;
    data.path=path;
    data.hoptotal=hoptotal;
    data.spaceK=spaceK;
    data.spaceR=spaceR;
    data.spaceT=spaceT;
    data.bandwidthK=bandwidthK;
    data.bandwidthR=bandwidthR;
    data.bandwidthT=bandwidthT;
    data.B=B;
    data.M=M;
    
    %% Generating Training Data
    imgData(:,:,:,index)=DataGenerator(data,para,image_layout);
    
    %% ILP solver
    result=ILP(para,data);
    
    %% Related Label
%     imgLabels(index,:)=result.allocations;
    
end

save(['../DataStore/imgData_' num2str(IMAGE) '.mat'],'imgData');
% save(['../DataStore/imgLabels_' num2str(IMAGE) '.mat'],'imgLabels');