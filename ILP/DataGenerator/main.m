%% preprocess
clear
clc

%% topology
[G,EdgeCloud,NormalRouter,AccessRouter,vertice_names]=SetNetTopo();
N=length(vertice_names);
for v=1:N
    eval([vertice_names{v},'=',num2str(v),';']);
end

%% parameter setting
% ID of mobile users  
flow=1:10; 
% caching cost per EC
alpha=1;
% transmission cost per hop
beta=0.05;
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
spacek=randi(3,size(flow));
% available space in EC

% total space in EC

% bandwidth requirement of flow

% available bandwidth in link

% total bandwidth in link

% relationship between node and link

% surfficiently large number