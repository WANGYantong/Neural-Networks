function [para,data] = Scenario(flow)

%% topology
[G,EdgeCloud,NormalRouter,AccessRouter,vertice_names]=SetNetTopo();
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

NF=length(flow);

NUMINDEX=1024+256;

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

save(['../DataStore/flow',num2str(flow(end)),'/network.mat'],'alpha','beta','gamma','hopcounter','hoptotal','B','G',...
    'EdgeCloud','AccessRouter');

para.graph=G;
para.EdgeCloud=EdgeCloud;
para.AccessRouter=AccessRouter;
para.NormalRouter=[GW,NormalRouter];
para.N=N;

data=cell(NUMINDEX,1);
    
rng(1);

HARDCORE=20;

for index=1:NUMINDEX
    % moving probability
    [probability,start_point]=SetMovProb(HARDCORE,length(AccessRouter));
    % space requirement of flow
    %     spaceK=randi([0,5],size(flow))*10+45;
    spaceK=randi([1,15],1,HARDCORE)*10;
    % available space in EC
    spaceR=[randi([7,8],size(EdgeCloud(1:3)))*50+101,randi([1,6],size(EdgeCloud(4:end)))*50+101];
%     spaceR=[ones(size(EdgeCloud(1:2)))*400+101,randi([1,2],size(EdgeCloud(3:end)))*100+51];  
    % bandwidth requirement of flow
    %     bandwidthK=randi([0,2],size(flow))*5+5;
    bandwidthK=randi([1,15],1,HARDCORE);
    % available bandwidth in link
    %     bandwidthR=randi([0,2],size(G.Edges.Weight))*10+80;
%     bandwidthR=randi([1,6],size(G.Edges.Weight))*10+40;
    bandwidthR=ones(size(G.Edges.Weight))*80;


    % packing parameters
    data{index}.flow=flow;
    data{index}.alpha=alpha;
    data{index}.beta=beta;
    data{index}.gamma=gamma;
    data{index}.probability=probability(1:NF,:);
    data{index}.startPoint=start_point(1:NF,:);
    data{index}.hopcounter=hopcounter;
    data{index}.path=path;
    data{index}.hoptotal=hoptotal;
    data{index}.spaceK=spaceK(1:NF);
    data{index}.spaceR=spaceR;
    data{index}.spaceT=spaceT;
    data{index}.bandwidthK=bandwidthK(1:NF);
    data{index}.bandwidthR=bandwidthR;
    data{index}.bandwidthT=bandwidthT;
    data{index}.B=B;
    data{index}.M=M;
        
end


end

