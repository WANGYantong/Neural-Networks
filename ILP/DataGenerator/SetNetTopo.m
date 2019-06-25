function [G,EdgeCloud,NormalRouter,AccessRouter,vertice_name] = SetNetTopo()

vertice_name={'GW','EC1','EC2','EC3','EC4','EC5','EC6','R1','R2','R3','R4','R5','R6','R7','R8'};

numNode=length(vertice_name);

for v=1:numNode
    eval([vertice_name{v},'=',num2str(v),';']);
end

EdgeCloud=EC1:EC6;
NormalRouter=R1:R8;
AccessRouter=NormalRouter;

s=repmat(GW:EC6,2,1);
s=reshape(s,1,[]);
t=EC1:R8;

weight=ones(size(s));

figure(1);

G=graph(s,t,weight,vertice_name);

% G=addedge(G,EC4,EC5,1);

h=plot(G,'NodeLabel',G.Nodes.Name);

highlight(h,GW,'NodeColor','g','Marker','p','MarkerSize',16);
highlight(h,EdgeCloud,'Marker','d','MarkerSize',12);
highlight(h,NormalRouter,'Marker','o','MarkerSize',12);
highlight(h,AccessRouter,'NodeColor','b');

h.XData=[8,4,12,2:4:14,1:2:15];
h.YData=[4,3*ones(1,2),2*ones(1,4),ones(1,8)];

title('Network Topology');

end
