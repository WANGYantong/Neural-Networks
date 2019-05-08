function probability = SetMovProb(NF,NA)

probability=zeros(NF,NA);

for ii=1:NF
    [base,label]=SetBase(NA);
    probability(ii,base)=1;
    if rand()>=0.5 % high moving desire
        if label==1
            probability(ii,base+1)=0.1*rand()+0.8;
            probability(ii,base)=probability(ii,base)-probability(ii,base+1);
        elseif label==3
            probability(ii,base-1)=0.1*rand()+0.8;
            probability(ii,base)=probability(ii,base)-probability(ii,base-1);
        else
            probability(ii,base-1)=0.05*rand()+0.4;
            probability(ii,base+1)=0.05*rand()+0.4;
            probability(ii,base)=probability(ii,base)...
                -probability(ii,base-1)-probability(ii,base+1);
        end
    else
        if label==1
            probability(ii,base+1)=0.09*rand()+0.01;
            probability(ii,base)=probability(ii,base)-probability(ii,base+1);
        elseif label==3
            probability(ii,base-1)=0.09*rand()+0.01;
            probability(ii,base)=probability(ii,base)-probability(ii,base-1);
        else
            probability(ii,base-1)=0.09*rand()+0.01;
            probability(ii,base+1)=0.09*rand()+0.01;
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
