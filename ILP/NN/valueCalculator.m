function cost = valueCalculator(img,label)

Net=load('../DataStore/network.mat');
layout=load('../DataStore/layout.mat');
Net.prob=(255-img(1:10,1:8))/255;
Net.sk=(255-img(1:10,9))*10/255;
Net.bk=(255-img(1:10,10))*10/255;
Net.SR=img(11:16,9)*50/255;
Net.BR=img(1:15,11)*100/255;

NF=length(label);
[~,NA,NE]=size(Net.B);

x=zeros(NF,NE);
z=zeros(NF,NA,NE);
for ii=1:NF
    x(ii,label(ii))=1;
    z(ii,Net.prob(ii,:)>0,label(ii))=1;
end

probability_z=repmat(Net.prob,[1,1,NE]);
hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);

cost=Net.alpha*sum(x,'all')+...
    Net.beta*sum(probability_z.*hopcounter_z.*z,'all')+...
    Net.beta*sum((1-sum(sum(probability_z.*z,3),2))*Net.hoptotal);

gamma=10;
cost=cost+gamma*(space_penalty(label,Net)+link_penalty(label,Net));

end

function spaceValue=space_penalty(label, Net)

NF=length(label);
flow_flag=ones(size(label));

for ii=1:NF
    if(Net.sk(ii) <= round(Net.SR(label(ii))))
        Net.SR(label(ii))=Net.SR(label(ii))-Net.sk(ii);
        flow_flag(ii)=0;
    end
end

spaceValue=sum(flow_flag*Net.sk);

end

function linkValue=link_penalty(label, Net)

NF=length(label);
[NL,NA,NE]=size(Net.B);

z=zeros(NF,NA,NE);
for ii=1:NF
    z(ii,Net.prob(ii,:)>0,label(ii))=1;
end

y=zeros(NF,NL);
for ii=1:NF
    for jj=1:NL
        if(sum(Net.B(jj,:,:).*z(ii,:,:),'all')>0)
            y(ii,jj)=1;
        end
    end
end

b_y=repmat(Net.bk,[1,NL]);
linkBuffer=sum(b_y.*y,1)-Net.BR';
linkValue=0;
for ii=1:length(linkBuffer)
    if linkBuffer(ii)>0
        linkValue=linkValue+linkBuffer(ii);
    end
end

end
