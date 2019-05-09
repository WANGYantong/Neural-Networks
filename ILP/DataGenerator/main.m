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
% moving probability
probability=SetMovProb(length(flow),length(AccessRouter));
% # of hops from AR to EC
hopcounter=zeros(length(AccessRouter),length(EdgeCloud));
path=cell(length(AccessRouter),length(EdgeCloud));
for ii=1:length(AccessRouter)
    for jj=1:length(EdgeCloud)
        [path{ii,jj},hopcounter(ii,jj)]=shortestpath(G,AccessRouter(ii),EdgeCloud(jj));
    end
end
% # of hops from AR to DataCenter
hoptotal=randi(5,size(flow))+10;
% space requirement of flow
spaceK=randi(3,size(flow));
% available space in EC
spaceR=randi(6,size(EdgeCloud))+12;
% total space in EC
spaceT=ones(size(EdgeCloud))*20;
% bandwidth requirement of flow
bandwidthK=randi(2,size(flow));
% available bandwidth in link
bandwidthR=ones(size(G.Edges.Weight))*18;
% total bandwidth in link
bandwidthT=bandwidthR/0.9;
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

%% ILP solver
result=ILP(para,data);
