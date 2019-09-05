function updateImage = imgUpdate(origImage,preallocation,layout)

NF=length(preallocation);
NE=length(layout.image_layout.space.y);
NL=length(layout.image_layout.bandwidth.y);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
NIMG=size(origImage,4); % number of images
assignment=[preallocation{1:NF}];

DONE=size(assignment,2);

updateImage=origImage;

for ii=1:NIMG   
    % prepare x for space update
    x=zeros(DONE,NE);
    for jj=1:DONE
        x(jj,assignment(ii,jj))=1;
    end
    % prepare y for bandwidth update
    z=zeros(DONE,NA,NE);
    for jj=1:DONE
            z(jj,Net.prob(jj,:)>0,assignment(ii,jj))=1;
    end
    y=zeros(DONE,NL);
    for jj=1:DONE
        for kk=1:NL
            if(sum(Net.B(kk,:,:).*z(jj,:,:),'all')>0)
                y(jj,kk)=1;
            end
        end
    end
    % update space
    
    % update link
    
end

end

