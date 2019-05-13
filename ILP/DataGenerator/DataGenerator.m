function matrix = DataGenerator(data,para)

NF=length(data.flow);
NL=length(para.graph.Edges.EndNodes);
NA=length(para.AccessRouter);
NE=length(para.EdgeCloud);

NUMCOL=max([NF,NL,NA,NE]);

matrix=ones(2*NF+4,NUMCOL)*255;

% mobile user starting point: black means current position
matrix(1:NF,1:size(data.startPoint,2))=255-data.startPoint*255;

% moving probability: deep color represents high probability
matrix(NF+1:2*NF,1:size(data.probability,2))=255-round(data.probability/1*255);

% space requirement: deep color means large space needed
normal=max(data.spaceK);
matrix(2*NF+1,1:size(data.spaceK,2))=255-round(data.spaceK/normal*255);

% edge cloud utilization: deep color means high utilizaiton(less available space)
matrix(2*NF+2,1:size(data.spaceR,2))=round(data.spaceR./data.spaceT*255);

% bandwidth requirement: deep color means large bandwidth needed
normal=max(data.bandwidthK);
matrix(2*NF+3,1:size(data.bandwidthK,2))=255-round(data.bandwidthK/normal*255);

% link utilization: deep color means high utilizaiton(less available bandwidth)
matrix(2*NF+4,1:size(data.bandwidthR',2))=round(data.bandwidthR./data.bandwidthT*255);

end

