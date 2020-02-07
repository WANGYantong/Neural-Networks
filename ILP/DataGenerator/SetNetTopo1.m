function [G,EdgeCloud,NormalRouter,AccessRouter,vertice_name] = SetNetTopo1()

vertice_name={'GW','R1','R2','R3','EC1','EC2','EC3','EC4','EC5','EC6',...
    'AR1','AR2','AR3','AR4','AR5','AR6','AR7'};

numNode=length(vertice_name);

for v=1:numNode
    eval([vertice_name{v},'=',num2str(v),';']);
end

EdgeCloud=EC1:EC6;
NormalRouter=[R1:R3, AR1:AR7];
AccessRouter=AR1:AR7;

s=repmat(GW:EC6,2,1);
s=reshape(s,1,[]);
t=[R1:EC1,EC1:EC3,EC1:AR7];

weight=ones(size(s));

figure(1);

G=graph(s,t,weight,vertice_name);

G=rmedge(G,R2,EC1);
G=rmedge(G,EC3,EC6);
G=addedge(G,EC2,EC6,1);
G=rmedge(G,EC4,AR2);
G=addedge(G,EC3,AR2,1);
G=addedge(G,EC3,EC4,1);

h=plot(G,'NodeLabel',G.Nodes.Name);

highlight(h,GW,'NodeColor',[0.4660 0.6740 0.1880],'Marker','p','MarkerSize',16);
highlight(h,EdgeCloud,'NodeColor',[0 0.4470 0.7410],'Marker','d','MarkerSize',12);
highlight(h,setdiff(NormalRouter,AccessRouter),'NodeColor',[0.3010 0.7450 0.9330],'Marker','o','MarkerSize',12);
highlight(h,AccessRouter,'NodeColor',[0.6350 0.0780 0.1840],'Marker','s','MarkerSize',12);

h.XData=[4,3,5,2:2:6,1:2:7,1:7];
h.YData=[5,4,4,3*ones(1,3),2*ones(1,4),ones(1,7)];

title('Network Topology');

end

