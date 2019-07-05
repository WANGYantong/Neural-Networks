function matrix = DataGenerator(data,para,layout)

% layout.opts=0 for Constants+Variables
% 1 for Variables; 2 for Centralized Variables;
% 3 for Value Normalization

if nargin ~= 3
    error('invalid input arguements');
end

switch layout.opts
    
    case 0  
        
        matrix=ones(layout.size)*255;
        
        % adjacency matrix of graph: black means link exist
        matrix(layout.graph.x,layout.graph.y)=255-full(adjacency(para.graph))*255;        
        % hop counter: deep color represents high value
        matrix(layout.hops.x,layout.hops.y)=255-data.hopcounter/max(data.hoptotal)*255;        
        % miss penalty
        matrix(layout.penalty.x,layout.penalty.y)=255-data.hoptotal/max(data.hoptotal)*255;
        
        % EC remaining space: deep color indicates less space available
%         matrix(layout.space.x,layout.space.y)=data.spaceR/max(data.spaceR)*255;
        matrix(layout.space.x,layout.space.y)=data.spaceR/50*255;
        % link remainng bandwidth
%         matrix(layout.bw.x,layout.bw.y)=data.bandwidthR'/max(data.bandwidthR)*255;
        matrix(layout.bw.x,layout.bw.y)=data.bandwidthR'/100*255;
        
        % mobile user starting point: black means current position
        matrix(layout.startpoint.x,layout.startpoint.y)=255-data.startPoint*255;
        % moving probability: deep color represents high probability
        matrix(layout.prob.x,layout.prob.y)=255-data.probability/1*255;
        % space requirement: deep color means large space needed
%         matrix(layout.sprq.x,layout.sprq.y)=255-data.spaceK'/max(data.spaceK)*255;   
        matrix(layout.sprq.x,layout.sprq.y)=255-data.spaceK'/10*255;    
        % bandwidth requirement: deep color means large bandwidth needed
%         matrix(layout.bwrq.x,layout.bwrq.y)=255-data.bandwidthK/max(data.bandwidthK)*255;
        matrix(layout.bwrq.x,layout.bwrq.y)=255-data.bandwidthK/10*255;

        
    case {1,2,3}
        
        matrix=ones(layout.size)*255;
        
        % EC remaining space: deep color indicates less space available
%         matrix(layout.space.x,layout.space.y)=data.spaceR/max(data.spaceR)*255;
        matrix(layout.space.x,layout.space.y)=data.spaceR/50*255;
        % link remainng bandwidth
%         matrix(layout.bw.x,layout.bw.y)=data.bandwidthR'/max(data.bandwidthR)*255;
        matrix(layout.bw.x,layout.bw.y)=data.bandwidthR'/100*255;
        
        % moving probability: deep color represents high probability
        matrix(layout.prob.x,layout.prob.y)=255-data.probability'/1*255;
        % space requirement: deep color means large space needed
%         matrix(layout.sprq.x,layout.sprq.y)=255-data.spaceK/max(data.spaceK)*255;
        matrix(layout.sprq.x,layout.sprq.y)=255-data.spaceK/10*255;  
        % bandwidth requirement: deep color means large bandwidth needed
%         matrix(layout.bwrq.x,layout.bwrq.y)=255-data.bandwidthK'/max(data.bandwidthK)*255;
        matrix(layout.bwrq.x,layout.bwrq.y)=255-data.bandwidthK'/10*255;
        
        
    case 4
        
        matrix=ones(layout.size)*255;
        
        % moving probability: deep color represents high probability
        matrix(layout.prob.x,layout.prob.y)=255-data.probability/1*255;
%         matrix(layout.prob.x,layout.prob.y)=data.probability;
        
        % space utilization: deep color indicates high utilization
        normal=transpose(transpose(1./data.spaceR)*data.spaceK);
        matrix(layout.space.x,layout.space.y)=255-normal/1*255;
%         matrix(layout.space.x,layout.space.y)=normal;
        
        % bandwidth utilization
        normal=transpose(1./data.bandwidthR*data.bandwidthK);
        matrix(layout.bandwidth.x,layout.bandwidth.y)=255-normal/1*255;
%         matrix(layout.bandwidth.x,layout.bandwidth.y)=normal;
              

    otherwise
        error('invalid argument layout.opts')
end

end

