function cost = valueCalculator(img,label,opt)
% opt for addtional decision variable y&z.
% opt.mode=0: generate y&z depend on x
% opt.mode=1: read y&z from external file

NF=length(label);

if not(iscategorical(label))
    label=categorical(label);
end  

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

[~,NA,NE]=size(Net.B);

% check the constraints satisfaction
% if not satisfy the 
while(1)
    [spaceValue,label]=spacePenalty(label,Net);
    if spaceValue ==0
        break;
    end
end

x=zeros(NF,NE);
for ii=1:NF
    if not(label(ii)==categorical(-1))
        x(ii,label(ii))=1;
    end
end

if opt.mode==0
    z=zeros(NF,NA,NE);
    for ii=1:NF
        if not(label(ii)==categorical(-1))
            z(ii,Net.prob(ii,:)>0,label(ii))=1;
        end
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
te=1./(1.0001-sum(Net.sk.*x,1));

% te(te<=0)=te(te<=0)-min(te)+100; % in case of minus value
% te(te<=0)=100;
% te(te==Inf)=100; % in case of infinity value

cost=Net.alpha*sum(x*te')+...
    Net.beta*sum(probability_z.*hopcounter_z.*z,'all')+...
    Net.gamma*sum((1-sum(sum(probability_z.*z,3),2))*Net.hoptotal);

delta=10;
cost=cost+delta*linkPenalty(label,Net,opt);
% cost=cost+delta*(spacePenalty(label,Net)+linkPenalty(label,Net,opt));

% cost=cost+gamma*(spacePenalty(label,Net));

% if linkPenalty(label,Net,opt)>0 || any(label==categorical(-1))
% % if spacePenalty(label,Net)>0
%     disp('infeasible');
% end

end

