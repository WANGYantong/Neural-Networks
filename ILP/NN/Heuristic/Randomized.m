function solution = Randomized(img)

Net=load('../DataStore/network.mat');
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

NF=length(sk);
NE=length(Net.EdgeCloud);

% implement the basic allocation (based on nearest
% EC from maximum moving probability)
original=FindEcForFlow(Net);

TIMES_HARDCODE=2000;
update=original;

for ii=1:TIMES_HARDCODE
    update(randi(NF))=categorical(randi(NE));
    valueUpdate=valueCalculator(img,update);
    valueOriginal=valueCalculator(img,original);
    if valueUpdate<valueOriginal
        original=update;
    else
        update=original;
    end
%     spaceFlagUpdate=space_check(update,Net);
%     linkFlagUpdate=link_check(update,Net);
%     spaceFlagOriginal=space_check(original,Net);
%     linkFlagOriginal=link_check(original,Net);
%     if spaceFlagUpdate && linkFlagUpdate
%         if ~spaceFlagOriginal || ~linkFlagOriginal
%             original=update; % previous assignment is invalid and find a legal one
%         else
%             valueFlag=value_compare(update,original,Net);
%             if valueFlag
%                 original=update; % previous is valid and find a better one
%             else
%                 update=original;
%             end
%         end
%     else
%         update=original;
%     end
end

solution=update;

end

function initial_point = FindEcForFlow(Net)

[~,I]=sort(Net.prob,2,'descend');
ar = Net.AccessRouter(I(:,1));

list_ec=Construct_EC_List(Net,ar);

initial_point=list_ec(:,1);

end

function list_ec = Construct_EC_List(Net,ar)

NF=length(Net.sk);
NE=length(Net.EdgeCloud);
list_cost=zeros(NF,NE);

for ii = 1:NF
    for jj = 1:NE
        [~,path_cost]=shortestpath(Net.G,ar(ii),Net.EdgeCloud(jj));
        list_cost(ii,jj)=path_cost;
    end
end

[~,list_ec]=sort(list_cost,2);

end

% function spaceFlag=space_check(label, Net)
% 
% NF=length(label);
% 
% if any(Net.SR) || any(Net.BR)    
%     flow_flag=zeros(size(label));
%     for ii=1:NF
%         if(Net.sk(ii) < round(Net.SR(label(ii))))
%             Net.SR(label(ii))=Net.SR(label(ii))-Net.sk(ii);
%             flow_flag(ii)=1;
%         end
%     end
% else
%     NE=size(Net.hopcounter,2);
%     x=zeros(NF,NE);
%     for ii=1:NF
%         x(ii,label(ii))=1;
%     end
%     flow_flag=sum(Net.sk.*x,1)<=1;
% end
% 
% spaceFlag=all(flow_flag);
% 
% end
% 
% function linkFlag=link_check(label, Net)
% 
% NF=length(label);
% [NL,NA,NE]=size(Net.B);
% 
% z=zeros(NF,NA,NE);
% for ii=1:NF
%     z(ii,Net.prob(ii,:)>0,label(ii))=1;
% end
% 
% y=zeros(NF,NL);
% for ii=1:NF
%     for jj=1:NL
%         if(sum(Net.B(jj,:,:).*z(ii,:,:),'all')>0)
%             y(ii,jj)=1;
%         end
%     end
% end
% 
% if any(Net.SR) || any(Net.BR)
%     b_y=repmat(Net.bk,[1,NL]);
%     linkFlag=all(sum(b_y.*y,1)<=Net.BR');
% else
%     b_y=Net.bk;
%     linkFlag=all(sum(b_y.*y,1)<=1);
% end
% 
% 
% end
% 
% function valueFlag=value_compare(labelUpdate,labelOriginal,Net)
% 
% NF=length(labelUpdate);
% [~,NA,NE]=size(Net.B);
% 
% xUpdate=zeros(NF,NE);
% xOriginal=zeros(NF,NE);
% for ii=1:NF
%     xUpdate(ii,labelUpdate(ii))=1;
%     xOriginal(ii,labelOriginal(ii))=1;
% end
% 
% zUpdate=zeros(NF,NA,NE);
% zOriginal=zeros(NF,NA,NE);
% for ii=1:NF
%     zUpdate(ii,Net.prob(ii,:)>0,labelUpdate(ii))=1;
%     zOriginal(ii,Net.prob(ii,:)>0,labelOriginal(ii))=1;
% end
% 
% probability_z=repmat(Net.prob,[1,1,NE]);
% hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
% hopcounter_z=repmat(hopcounter_z,[NF,1]);
% hopcounter_z=reshape(hopcounter_z,NF,NA,NE);
% 
% costUpdate=Net.alpha*sum(xUpdate,'all')+...
%     Net.beta*sum(probability_z.*hopcounter_z.*zUpdate,'all')+...
%     Net.beta*sum((1-sum(sum(probability_z.*zUpdate,3),2)).*Net.hoptotal);
% 
% costOriginal=Net.alpha*sum(xOriginal,'all')+...
%     Net.beta*sum(probability_z.*hopcounter_z.*zOriginal,'all')+...
%     Net.beta*sum((1-sum(sum(probability_z.*zOriginal,3),2)).*Net.hoptotal);
% 
% valueFlag=costUpdate<costOriginal;
% 
% end
