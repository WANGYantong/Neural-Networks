function [final_state,final_score] = HillClimbing2(img,init_state,score)

NF=length(init_state);
NE=length(score)/NF;

scoreRe=reshape(score,[NE,NF])';
scoreRe(scoreRe<1e-4)=0;

opt.mode=0;
init_score=valueCalculator2(img,init_state,opt);
    
succ_state=cell(NF,1);
succ_index=2*ones(NF,1);
succ_score=zeros(NF,1);

while(1)
 
    for ii=1:NF
        [succ_state{ii},update]=FindSucc2(init_state,scoreRe(ii,:),succ_index(ii),ii);
        
        succ_state{ii}'
        
        if update==1
            succ_score(ii)=valueCalculator2(img,succ_state{ii},opt);
            
            succ_score(ii)'
            
        else
            succ_score(ii)=init_score;
        end
    end
    
    if init_score<min(succ_score)
        break;
    else
        [init_score,index]=min(succ_score);
        init_state=succ_state{index};
        succ_index(index)=succ_index(index)+1;
    end

end

final_state=init_state;
final_score=init_score;

end

function [succ_state,update]=FindSucc2(init_state, score, succ_index,index)

succ_state=init_state;
update=0;
[~,II]=maxk(score, succ_index);

if II(end)>0
    succ_state(index)=categorical(II(end));
    update=1;
end

end

