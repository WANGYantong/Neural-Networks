function matrix = DataGenerator(data,para,opts)

% opts=0 sparse image; opts=1 dense image;
% opts=2 combine moving probability & divide utilization into
% remaining+total
% opts=3 add hop counter matrix & remove total space+bw 

if nargin == 2
    opts=0;
end

NF=length(data.flow);
NL=length(para.graph.Edges.EndNodes);
NA=length(para.AccessRouter);
NE=length(para.EdgeCloud);

NUMCOL=max([NF,NL,NA,NE]);

switch opts
    
    case 0  
        
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
        
    case 1
        
        matrix=ones(21,10)*255;
        
        % mobile user starting point: black means current position
        matrix(1:10,1:8)=255-data.startPoint*255;
        
        % moving probability: deep color represents high probability
        matrix(11:20,1:8)=255-round(data.probability/1*255);
        
        % space requirement: deep color means large space needed
        normal=max(data.spaceK);
        matrix(1:10,9)=255-round(data.spaceK'/normal*255);
        
        % edge cloud utilization: deep color means high utilizaiton(less available space)
        matrix(16:21,10)=round(data.spaceR'./data.spaceT'*255);
        
        % bandwidth requirement: deep color means large bandwidth needed
        normal=max(data.bandwidthK);
        matrix(11:20,9)=255-round(data.bandwidthK'/normal*255);
        
        % link utilization: deep color means high utilizaiton(less available bandwidth)
        matrix(1:15,10)=round(data.bandwidthR./data.bandwidthT*255);
        
    case 2
        
        matrix=ones(15,12)*255;
        
        % moving probability: deep color represents high probability
        matrix(1:10,1:8)=255-round(data.probability/1*255);
        
        % space requirement: deep color means large space needed
        normal=max(data.spaceK);  
        matrix(1:10,9)=255-round(data.spaceK'/normal*255);
        
        % bandwidth requirement: deep color means large bandwidth needed
        normal=max(data.bandwidthK);
        matrix(1:10,10)=255-round(data.bandwidthK'/normal*255);
        
        % edge cloud remaining space
        normal=max(data.spaceR);
        matrix(11,1:6)=round(data.spaceR/normal*255);     
        
        % edge cloud total space
        normal=max(data.spaceT);
        matrix(12,1:6)=round(data.spaceT/normal*255);     
        
        % link remaing bandwidth
        normal=max(data.bandwidthR);
        matrix(1:15,11)=round(data.bandwidthR/normal*255);
        
        % link total bandwidth
        normal=max(data.bandwidthT);
        matrix(1:15,12)=round(data.bandwidthT./normal*255);
        
    case 3
        
        matrix=ones(16,11)*255;
        
        % moving probability: deep color represents high probability
        matrix(1:10,1:8)=255-data.probability*255;
        
        % hop counter: deep color represents high value
        matrix(11:16,1:8)=255-data.hopcounter'/max(data.hoptotal)*255;
        
        % space requirement: deep color means large space needed
%         normal=max(data.spaceK);
        normal=10;
        matrix(1:10,9)=255-data.spaceK'/normal*255;
        
        % bandwidth requirement: deep color means large bandwidth needed
        normal=10;
        matrix(1:10,10)=255-data.bandwidthK'/normal*255;
        
        % edge cloud remaining space
        normal=50;
        matrix(11:16,9)=data.spaceR'/normal*255;

        % link remaing bandwidth
        normal=100;
        matrix(1:15,11)=data.bandwidthR/normal*255;
        
end

end

