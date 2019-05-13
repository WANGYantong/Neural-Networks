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
alpha=1;
% transmission cost per hop
beta=0.25;

NF=length(flow);
NA=length(AccessRouter);
NE=length(EdgeCloud);
NL=length(G.Edges.EndNodes);

NUMINDEX=10000;

imgData=zeros(2*NF+4, max([NF,NL,NA,NE]),1,NUMINDEX);
imgLabels=zeros(NUMINDEX,NF);

for INDEX=1:NUMINDEX
    
    rng(INDEX);
    
    % moving probability
    [probability,start_point]=SetMovProb(length(flow),length(AccessRouter));
    % # of hops from AR to EC
    hopcounter=zeros(length(AccessRouter),length(EdgeCloud));
    path=cell(length(AccessRouter),length(EdgeCloud));
    for ii=1:length(AccessRouter)
        for jj=1:length(EdgeCloud)
            [path{ii,jj},hopcounter(ii,jj)]=shortestpath(G,AccessRouter(ii),EdgeCloud(jj));
        end
    end
    % # of hops from AR to DataCenter
    hoptotal=ones(size(flow))*15;
    % space requirement of flow
    spaceK=randi(10,size(flow));
    % available space in EC
    spaceR=randi(40,size(EdgeCloud))+10;
    % total space in EC
    spaceT=ones(size(EdgeCloud))*50;
    % bandwidth requirement of flow
    bandwidthK=randi(10,size(flow));
    % available bandwidth in link
    bandwidthR=randi(50,size(G.Edges.Weight))+50;
    % total bandwidth in link
    bandwidthT=ones(size(G.Edges.Weight))*100;
    % relationship between node and link
    B=GetPathLinkRel(G,"undirected",path,length(AccessRouter),length(EdgeCloud));
    % surfficiently large number
    M=1000;
    
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
    imgData(:,:,:,INDEX)=DataGenerator(data,para);
    
    %% ILP solver
    result=ILP(para,data);
    
    %% Related Label
    imgLabels(INDEX,:)=result.allocations;
    
end

save('../DataStore/imgData.mat','imgData');
save('../DataStore/imgLabels.mat','imgLabels');