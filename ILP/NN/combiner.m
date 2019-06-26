function labelUpdate = combiner(img,labelOriginal,score,opts)

% if nargin < 3
%     opts=0; %  totally randomized
% else
%     opts=2; % based on prediction score
% end

Net=load('../DataStore/network.mat');
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

NF=length(labelOriginal);
NE=length(score)/NF;

switch opts
    
    case 0
        
        TIMES_HARDCODE=1000;
        
        labelUpdate=labelOriginal;
        for ii=1:TIMES_HARDCODE
            labelUpdate(randi(NF))=categorical(randi(NE));
            spaceFlagUpdate=space_check(labelUpdate,Net);
            linkFlagUpdate=link_check(labelUpdate,Net);
            spaceFlagOriginal=space_check(labelOriginal,Net);
            linkFlagOriginal=link_check(labelOriginal,Net);
            if spaceFlagUpdate && linkFlagUpdate
                if ~spaceFlagOriginal || ~linkFlagOriginal
                    labelOriginal=labelUpdate; % previous assignment is invalid and find a legal one
                else
                    valueFlag=value_compare(labelUpdate,labelOriginal,Net);
                    if valueFlag
                        labelOriginal=labelUpdate; % previous is valid and find a better one
                    else
                        labelUpdate=labelOriginal;
                    end
                end
            else
                labelUpdate=labelOriginal;
            end
        end        
        
    case 1
        scoreRe=reshape(score,[NE,NF])';
        [row,col]=find(scoreRe>=0.001);
        [row,ind]=sort(row);
        col=categorical(col(ind));
        
        labelUpdate=labelOriginal;
        for ii=1:length(row)  % considering modify depending on the value of score desendly
            if labelUpdate(row(ii))~=col(ii)
                labelUpdate(row(ii))=col(ii);
                spaceFlagUpdate=space_check(labelUpdate,Net);
                linkFlagUpdate=link_check(labelUpdate,Net);
                spaceFlagOriginal=space_check(labelOriginal,Net);
                linkFlagOriginal=link_check(labelOriginal,Net);
                if spaceFlagUpdate && linkFlagUpdate   
                    if ~spaceFlagOriginal || ~linkFlagOriginal
                        labelOriginal=labelUpdate; % previous assignment is invalid and find a legal one
                    else
                        valueFlag=value_compare(labelUpdate,labelOriginal,Net);
                        if valueFlag
                            labelOriginal=labelUpdate; % previous is valid and find a better one
                        else
                            labelUpdate=labelOriginal;
                        end
                    end
                else
                    labelUpdate=labelOriginal;
                end
            end
        end   
        
    case 2
        scoreRe=reshape(score,[NE,NF])';
        [row,col]=find(scoreRe>=0.0001);
        value=zeros(size(row));
        for ii=1:length(row)
            value(ii)=scoreRe(row(ii),col(ii));
        end
        [~,ind]=sort(value,'descend');
        row=row(ind);
        col=categorical(col(ind));
        
        labelUpdate=labelOriginal;
        for ii=1:length(row)  
            if labelUpdate(row(ii))~=col(ii)
                labelUpdate(row(ii))=col(ii);
                spaceFlagUpdate=space_check(labelUpdate,Net);
                linkFlagUpdate=link_check(labelUpdate,Net);
                spaceFlagOriginal=space_check(labelOriginal,Net);
                linkFlagOriginal=link_check(labelOriginal,Net);
                if spaceFlagUpdate && linkFlagUpdate   
                    if ~spaceFlagOriginal || ~linkFlagOriginal
                        labelOriginal=labelUpdate; % previous assignment is invalid and find a legal one
                    else
                        valueFlag=value_compare(labelUpdate,labelOriginal,Net);
                        if valueFlag
                            labelOriginal=labelUpdate; % previous is valid and find a better one
                        else
                            labelUpdate=labelOriginal;
                        end
                    end
                else
                    labelUpdate=labelOriginal;
                end
            end
        end   
end

end

function spaceFlag=space_check(label, Net)

NF=length(label);

if any(Net.SR) || any(Net.BR)    
    flow_flag=zeros(size(label));
    for ii=1:NF
        if(Net.sk(ii) < round(Net.SR(label(ii))))
            Net.SR(label(ii))=Net.SR(label(ii))-Net.sk(ii);
            flow_flag(ii)=1;
        end
    end
else
    NE=size(Net.hopcounter,2);
    x=zeros(NF,NE);
    for ii=1:NF
        x(ii,label(ii))=1;
    end
    flow_flag=sum(Net.sk.*x,1)<=1;
end

spaceFlag=all(flow_flag);

end

function linkFlag=link_check(label, Net)

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
else
    b_y=Net.bk;
end
linkFlag=all(sum(b_y.*y,1)<=Net.BR');

end

function valueFlag=value_compare(labelUpdate,labelOriginal,Net)

NF=length(labelUpdate);
[~,NA,NE]=size(Net.B);

xUpdate=zeros(NF,NE);
xOriginal=zeros(NF,NE);
for ii=1:NF
    xUpdate(ii,labelUpdate(ii))=1;
    xOriginal(ii,labelOriginal(ii))=1;
end

zUpdate=zeros(NF,NA,NE);
zOriginal=zeros(NF,NA,NE);
for ii=1:NF
    zUpdate(ii,Net.prob(ii,:)>0,labelUpdate(ii))=1;
    zOriginal(ii,Net.prob(ii,:)>0,labelOriginal(ii))=1;
end

probability_z=repmat(Net.prob,[1,1,NE]);
hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);

costUpdate=Net.alpha*sum(xUpdate,'all')+...
    Net.beta*sum(probability_z.*hopcounter_z.*zUpdate,'all')+...
    Net.beta*sum((1-sum(sum(probability_z.*zUpdate,3),2)).*Net.hoptotal);

costOriginal=Net.alpha*sum(xOriginal,'all')+...
    Net.beta*sum(probability_z.*hopcounter_z.*zOriginal,'all')+...
    Net.beta*sum((1-sum(sum(probability_z.*zOriginal,3),2)).*Net.hoptotal);

valueFlag=costUpdate<costOriginal;

end
