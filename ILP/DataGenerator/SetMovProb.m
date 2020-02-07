function [probability,start_point] = SetMovProb(NF,NA)

probability=zeros(NF,NA);
start_point=zeros(NF,NA);

for ii=1:NF
    [base,label]=SetBase(NA);
    probability(ii,base)=1;
    start_point(ii,base)=1;
    if rand()>=0.5 % high moving desire
        if label==1
            probability(ii,base+1)=0.3*rand()+0.6;
            probability(ii,base)=probability(ii,base)-probability(ii,base+1);
        elseif label==3
            probability(ii,base-1)=0.3*rand()+0.6;
            probability(ii,base)=probability(ii,base)-probability(ii,base-1);
        else
            probability(ii,base-1)=0.1*rand()+0.35;
            probability(ii,base+1)=0.1*rand()+0.35;
            probability(ii,base)=probability(ii,base)...
                -probability(ii,base-1)-probability(ii,base+1);
        end
    else
        if label==1
            probability(ii,base+1)=0.2*rand()+0.2;
            probability(ii,base)=probability(ii,base)-probability(ii,base+1);
        elseif label==3
            probability(ii,base-1)=0.2*rand()+0.2;
            probability(ii,base)=probability(ii,base)-probability(ii,base-1);
        else
            probability(ii,base-1)=0.1*rand()+0.15;
            probability(ii,base+1)=0.1*rand()+0.15;
            probability(ii,base)=probability(ii,base)...
                -probability(ii,base-1)-probability(ii,base+1);
        end
    end
end

end

function [base,label]=SetBase(NA)

base=randi([1,NA]);

switch base
    case 1
        label=1;
    case NA
        label=3;
    otherwise
        label=2;
end

end
