function [G,EdgeCloud,NormalRouter,AccessRouter,vertice_name] = SetNetTopo()

vertice_name={'GW','R1','R2','R3','EC1','EC2','EC3','EC4','EC5','EC6',...
    'EC7','AR1','AR2','AR3','AR4','AR5','AR6','AR7','AR8'};

numNode=length(vertice_name);

for v=1:numNode
    eval([vertice_name{v},'=',num2str(v),';']);
end

EdgeCloud=EC1:EC7;
NormalRouter=[R1:R3, AR1:AR8];
AccessRouter=AR1:AR8;

s=repmat(GW:EC7,2,1);
s=reshape(s,1,[]);
t=[R1:EC1,EC1:EC3,EC1:AR8];

weight=ones(size(s));

figure(1);

G=graph(s,t,weight,vertice_name);

G=rmedge(G,R2,EC1);
G=addedge(G,R2,EC3,1);
G=rmedge(G,EC2,EC3);
G=rmedge(G,R3,EC3);
G=addedge(G,R3,EC4,1);
G=rmedge(G,EC1,EC3);
G=addedge(G,EC1,EC4,1);
G=rmedge(G,EC2,EC4);
G=addedge(G,EC2,EC6,1);
G=addedge(G,EC2,EC7,1);
G=rmedge(G,EC3,EC6);
G=rmedge(G,EC3,EC7);
G=addedge(G,EC3,AR8,1);
G=addedge(G,EC4,EC5,1);
G=rmedge(G,EC5,AR4);
G=rmedge(G,EC6,AR6);
G=addedge(G,EC6,AR4,1);
G=rmedge(G,EC7,AR8);
G=addedge(G,EC7,AR6,1);

h=plot(G,'NodeLabel',G.Nodes.Name);

highlight(h,GW,'NodeColor',[0.4660 0.6740 0.1880],'Marker','p','MarkerSize',16);
highlight(h,EdgeCloud,'NodeColor',[0 0.4470 0.7410],'Marker','d','MarkerSize',12);
highlight(h,setdiff(NormalRouter,AccessRouter),'NodeColor',[0.3010 0.7450 0.9330],'Marker','o','MarkerSize',12);
highlight(h,AccessRouter,'NodeColor',[0.6350 0.0780 0.1840],'Marker','s','MarkerSize',12);

h.XData=[6,4,8,repmat([2,5,8,11],1,2),1:2:9,10,12,13];
h.YData=[5,4,4,3*ones(1,4),2*ones(1,4),ones(1,8)];

title('Network Topology');

end
