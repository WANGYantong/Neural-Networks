function [para,data] = Scenario(HARDCORE)

%% topology
[G,EdgeCloud,NormalRouter,AccessRouter,vertice_names]=SetNetTopo1();
N=length(vertice_names);
for v=1:N
    eval([vertice_names{v},'=',num2str(v),';']);
end

%% setting parameter
% caching cost per EC
alpha=0.5;
% transmission cost per hop
beta=1;
% cache miss penalty
gamma=1;

NUMINDEX=128+128+1024;

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
M=1e6;

para.graph=G;
para.EdgeCloud=EdgeCloud;
para.AccessRouter=AccessRouter;
para.NormalRouter=[GW,NormalRouter];
para.N=N;

data=cell(NUMINDEX,1);
    
rng(1);

for index=1:NUMINDEX
%     rng(index);
    % moving probability
    [probability,start_points]=SetMovProb(HARDCORE,length(AccessRouter));
    % space requirement of flow
    spaceK=randi([1,20],1,HARDCORE)*5;
%     spaceK=[randi([10,20],1,HARDCORE/2)*10,randi([5,15],1,HARDCORE/2)*10];
    % available space in EC
%     spaceR=[randi([6,8],size(EdgeCloud(1:2)))*50+100,randi([1,5],size(EdgeCloud(3:end)))*50+100];
%     spaceR=[ones(size(EdgeCloud(1:2)))*400+101,randi([1,4],size(EdgeCloud(3:end)))*100+51];  
    spaceR=[randi([4,5],size(EdgeCloud(1:2)))*100+51,randi([3,5],size(EdgeCloud(3:end)))*50+51];
    % bandwidth requirement of flow
%     bandwidthK=randi([0,2],1,HARDCORE)*5+5;
    bandwidthK=randi([1,8],1,HARDCORE)*5;
    % available bandwidth in link
%     bandwidthR=randi([0,2],size(G.Edges.Weight))*10+80;
%     bandwidthR=randi([1,6],size(G.Edges.Weight))*10+40;
    bandwidthR=ones(size(G.Edges.Weight))*101;


    % packing parameters
    data{index}.alpha=alpha;
    data{index}.beta=beta;
    data{index}.gamma=gamma;
    data{index}.probability=probability;
    data{index}.startPoints=start_points;
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


end

