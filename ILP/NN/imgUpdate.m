function updateImage = imgUpdate(origImage,preallocation,layout)

NF=length(layout.image_layout.space.x);
NE=length(layout.image_layout.space.y);
NL=length(layout.image_layout.bandwidth.y);
NA=length(layout.image_layout.prob.y);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
NIMG=size(origImage,4); % number of images
assignment=[preallocation{1:NF}];

updateImage=origImage;

DONE=size(assignment,2);
if DONE == NF
    return;
end

for ii=1:NIMG
    
    [Net.prob,Net.sk,Net.bk,Net.SR,Net.BR]=imageDecoding(updateImage(:,:,:,ii));
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
    usedSpace=sum(Net.sk(1:DONE,:).*x,1);
    updatedSpace=Net.sk(DONE+1:end,:)./(1-usedSpace);
    updateImage(layout.image_layout.space.x(DONE+1:end),...
        layout.image_layout.space.y,:,ii)=updatedSpace;
    % update link
    usedBandwidth=sum(Net.bk(1:DONE,:).*y,1);
    updatedBandwidth=Net.bk(DONE+1:end,:)./(1-usedBandwidth);
    updateImage(layout.image_layout.bandwidth.x(DONE+1:end),...
        layout.image_layout.bandwidth.y,:,ii)=updatedBandwidth;
end

end

