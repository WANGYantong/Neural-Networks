function layout = ImageEncoding(N,NF,NE,NA,NL,IMAGE)

if nargin ~= 6
    error('Invalid input argument');
end

layout.opts=IMAGE;

switch IMAGE
    
    case 0
        % Constants + Variables
        layout.size=[N+1+NF,max([N+NE+1,NE+NL,2*NA+2])];
        
        layout.graph.x=1:N;
        layout.graph.y=1:N;
        layout.hops.x=1:NA;
        layout.hops.y=layout.graph.y(end)+1:layout.graph.y(end)+NE;
        layout.penalty.x=layout.hops.x;
        layout.penalty.y=layout.hops.y(end)+1;
        
        layout.space.x=layout.graph.x(end)+1;
        layout.space.y=1:NE;
        layout.bw.x=layout.space.x;
        layout.bw.y=layout.space.y(end)+1:layout.space.y(end)+NL;
        
        layout.startpoint.x=layout.space.x(end)+1:layout.space.x(end)+NF;
        layout.startpoint.y=1:NA;
        layout.prob.x=layout.startpoint.x;
        layout.prob.y=layout.startpoint.y(end)+1:layout.startpoint.y(end)+NA;
        layout.sprq.x=layout.startpoint.x;
        layout.sprq.y=layout.prob.y(end)+1;
        layout.bwrq.x=layout.startpoint.x;
        layout.bwrq.y=layout.sprq.y(end)+1;      
  
        
    case 1
        % Variables
        layout.size=[4+NA,max([NF,NE,NL])];
        
        layout.space.x=1;
        layout.space.y=1:NE;
        layout.bw.x=layout.space.x(end)+1;
        layout.bw.y=1:NL;
        
        layout.prob.x=layout.bw.x(end)+1:layout.bw.x(end)+NA;
        layout.prob.y=1:NF;
        layout.sprq.x=layout.prob.x(end)+1;
        layout.sprq.y=1:NF;
        layout.bwrq.x=layout.sprq.x(end)+1;
        layout.bwrq.y=1:NF;
    
        
    case {2,3}
        % Centralized Variables
        layout.size=[4+NA,max([NF,NE,NL])];
        Col=layout.size(2);
        
        layout.space.x=1;
        layout.space.y=1:NE;
        if length(layout.space.y)<Col
            layout.space.y=floor(Col/2)-floor(layout.space.y/2):floor(Col/2)-floor(layout.space.y/2)+NE;
        end
        layout.bw.x=layout.space.x(end)+1;
        layout.bw.y=1:NL;
        if length(layout.bw.y)<Col
            layout.bw.y=floor(Col/2)-floor(layout.bw.y/2):floor(Col/2)-floor(layout.bw.y/2)+NL;
        end
        
        layout.prob.x=layout.bw.x(end)+1:layout.bw.x(end)+NA;
        layout.prob.y=1:NF;
        if length(layout.prob.y)<Col
            layout.prob.y=floor(Col/2)-floor(layout.prob.y/2):floor(Col/2)-floor(layout.prob.y/2)+NF;
        end
        layout.sprq.x=layout.prob.x(end)+1;
        layout.sprq.y=1:NF;
        if length(layout.sprq.y)<Col
            layout.sprq.y=floor(Col/2)-floor(layout.sprq.y/2):floor(Col/2)-floor(layout.sprq.y/2)+NF;
        end
        layout.bwrq.x=layout.sprq.x(end)+1;
        layout.bwrq.y=1:NF;
        if length(layout.bwrq.y)<Col
            layout.bwrq.y=floor(Col/2)-floor(layout.bwrq.y/2):floor(Col/2)-floor(layout.bwrq.y/2)+NF;
        end
        
        if IMAGE == 3
            layout.space.x=1;
            layout.sprq.x=layout.space.x(end)+1;
            layout.bw.x=layout.sprq.x(end)+1;
            layout.bwrq.x=layout.bw.x(end)+1;
            layout.prob.x=layout.bwrq.x(end)+1:layout.bwrq.x(end)+NA;
        end
            
        
    case 4
        % Value Normalization
        layout.size=[NF,NA+NE+NL];
        
        layout.prob.x=1:NF;
        layout.prob.y=1:NA;
        layout.space.x=layout.prob.x;
        layout.space.y=layout.prob.y(end)+1:layout.prob.y(end)+NE;
        layout.bandwidth.x=layout.prob.x;
        layout.bandwidth.y=layout.space.y(end)+1:layout.space.y(end)+NL;
        
        
    otherwise
        error('invalid argument IMAGE');
end

end

