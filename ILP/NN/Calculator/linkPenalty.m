function [linkValue,link_flag]=linkPenalty(Net, opt)

NF=opt.NF;
NL=opt.NL;
NA=opt.NA;
NE=opt.NE;

if opt.mode==0
    z=opt.z;
    y=zeros(NF,NL);
    for jj=1:NL
        [ind1,ind2]=find(opt.B_fold{jj});
        if isempty(ind1)
            continue;
        end
        indices=sub2ind([NA,NE],ind1,ind2);
        for ii=1:NF
            if any(z(ii,indices))
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

