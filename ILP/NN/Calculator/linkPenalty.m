function [linkValue,link_flag]=linkPenalty(label, Net, opt)

NF=length(label);
[NL,NA,NE]=size(Net.B);

if opt.mode==0
    z=zeros(NF,NA,NE);
    for ii=1:NF
        if not(label(ii)==categorical(-1))
            z(ii,Net.prob(ii,:)>0,label(ii))=1;
        end
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

