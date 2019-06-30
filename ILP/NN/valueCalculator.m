function cost = valueCalculator(img,label)

Net=load('../DataStore/network.mat');
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

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
NE=size(Net.hopcounter,2);
x=zeros(NF,NE);
for ii=1:NF
    x(ii,label(ii))=1;
end
    
if any(Net.SR) || any(Net.BR)
    s_x=repmat(Net.sk,[1,NE]);
    flow_flag=sum(s_x.*x,1)<=Net.SR';
else
    s_x=Net.sk;
    flow_flag=sum(s_x.*x,1)<=1;
end

spaceValue=sum(1-flow_flag);

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

if any(Net.SR) || any(Net.BR)
    b_y=repmat(Net.bk,[1,NL]);
    link_flag=sum(b_y.*y,1)<=Net.BR';
else
    b_y=Net.bk;
    link_flag=sum(b_y.*y,1)<=1;
end

linkValue=sum(1-link_flag);

end
