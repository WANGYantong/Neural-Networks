function cost = valueCalculator(Net,label,opt)
% opt for addtional decision variable y&z.
% opt.mode=0: generate y&z depend on x
% opt.mode=1: read y&z from external file

% if not(iscategorical(label))
%     label=categorical(label);
% end  

NF=opt.NF;
NE=opt.NE;
NA=opt.NA;
% check the constraints satisfaction
% if not satisfy the 
% while(1)
%     [spaceValue,label]=spacePenalty(label,Net);
%     if spaceValue ==0
%         break;
%     end
% end

indicator=not(label==opt.unvalid);

x=zeros(NF,NE);
row=(1:NF).*indicator;
row(row==0)=[];
col=label.*indicator;
col(col==0)=[];
indices=sub2ind(size(x),row,col);
x(indices)=1;
% for ii=1:NF
%     if indicator(ii)
%         x(ii,label(ii))=1;
%     end
% end

if opt.mode==0
    z=zeros(NF,NA,NE);
    
    Net.prob(indicator==0,:)=0;
    [ind1,ind2]=find(Net.prob);
    ind3=label(ind1);
    indices=sub2ind(size(z),ind1,ind2,ind3');
    z(indices)=1;
%     for ii=1:NF
%          if indicator(ii)
%             z(ii,Net.prob(ii,:)>0,label(ii))=1;
%          end
%     end
    opt.z=z;
else
   z=opt.z;
end

probability_z=repmat(Net.prob,[1,1,NE]);
hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);

x=round(x);
z=round(z);
te=1./(1-sum(Net.sk.*x,1));

te(te<=0)=te(te<=0)-min(te)+100; % in case of minus value
% te(te<=0)=100;
te(te==Inf)=100; % in case of infinity value

cost=Net.alpha*sum(x*te')+...
    Net.beta*sum(probability_z.*hopcounter_z.*z,'all')+...
    Net.gamma*sum((1-sum(sum(probability_z.*z,3),2))*Net.hoptotal);

delta=20;
[linkValue,link_flag]=linkPenalty(Net, opt);
cost=cost+delta*linkValue;
% cost=cost+delta*(spacePenalty(label,Net)+linkPenalty(label,Net,opt));

% cost=cost+gamma*(spacePenalty(label,Net));

% if any(link_flag)==0 || any(label==categorical(-1))
% if linkValue>0 || any(label==categorical(-1))
% % % if spacePenalty(label,Net)>0
%     fprintf('\n%d', sum(1-link_flag)+sum(label==categorical(-1)));
% end

end

