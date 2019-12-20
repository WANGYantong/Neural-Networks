function [spaceValue,label]=spacePenalty(label, Net, opt)

NF=length(label);
NE=size(Net.hopcounter,2);
x=zeros(NF,NE);
for ii=1:NF
%     if isundefined(label(ii))
%         disp(label);
%     end
    if not(label(ii)==opt.unvalid)
        x(ii,label(ii))=1;
    end
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

if spaceValue>0
    collision=x(:,not(flow_flag));
    pos=find(collision(:,1));
    label(pos(end))=opt.unvalid;
end

end

