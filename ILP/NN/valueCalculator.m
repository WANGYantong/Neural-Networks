function cost = valueCalculator(img,label,opt)
% opt for addtional decision variable y&z.
% opt.mode=0: generate y&z depend on x
% opt.mode=1: read y&z from external file

global flow;

Net=load(['../DataStore/',num2str(flow(end)),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

NF=length(label);
[~,NA,NE]=size(Net.B);

x=zeros(NF,NE);
for ii=1:NF
    x(ii,label(ii))=1;    
end

if opt.mode==0
    z=zeros(NF,NA,NE);
    for ii=1:NF
        z(ii,Net.prob(ii,:)>0,label(ii))=1;
    end
else
   z=opt.z;
end

probability_z=repmat(Net.prob,[1,1,NE]);
hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);

x=round(x);
z=round(z);
cost=Net.alpha*sum(x,'all')+...
    Net.beta*sum(probability_z.*hopcounter_z.*z,'all')+...
    Net.beta*sum((1-sum(sum(probability_z.*z,3),2))*Net.hoptotal);

gamma=20;
cost=cost+gamma*(space_penalty(label,Net)+link_penalty(label,Net,opt));

if (space_penalty(label,Net)+link_penalty(label,Net,opt))>0
    disp('infeasible');
end

end

function spaceValue=space_penalty(label, Net)

NF=length(label);
NE=size(Net.hopcounter,2);
x=zeros(NF,NE);
for ii=1:NF
    x(ii,label(ii))=1;
end
    
x=round(x);
epsilon=0.00001;
if any(Net.SR) || any(Net.BR)
    s_x=repmat(Net.sk,[1,NE]);
    flow_flag=sum(s_x.*x,1)<=(1+epsilon)*Net.SR';
    spaceValue=sum(sum(s_x.*x,1).*(1-flow_flag)./Net.SR');
else
    s_x=Net.sk;
    flow_flag=sum(s_x.*x,1)-epsilon<=1;
    spaceValue=sum(sum(s_x.*x,1).*(1-flow_flag));
end

end

function linkValue=link_penalty(label, Net, opt)

NF=length(label);
[NL,NA,NE]=size(Net.B);

if opt.mode==0
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
else
    y=opt.y;
end

y=round(y);
epsilon=0.00001;
if any(Net.SR) || any(Net.BR)
    b_y=repmat(Net.bk,[1,NL]);
    link_flag=sum(b_y.*y,1)<=(1+epsilon)*Net.BR';
    linkValue=sum(sum(b_y.*y,1).*(1-link_flag)./Net.BR');
else
    b_y=Net.bk;
    link_flag=sum(b_y.*y,1)-epsilon<=1;
    linkValue=sum(sum(b_y.*y,1).*(1-link_flag));
end

end
