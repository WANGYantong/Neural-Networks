function [final_state,final_score] = HillClimbing(img,init_state,score)

init_state=double(string(init_state)); % convert from catgorical to int

NF=length(init_state);
NE=length(score)/NF;

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

scoreRe=reshape(score,[NE,NF])';
scoreRe(scoreRe<1e-4)=0;

[opt.NL,opt.NA,opt.NE]=size(Net.B);
opt.mode=0;
opt.NF=NF;
opt.unvalid=-1;
B_fold=cell(opt.NL,1);
for ii=1:opt.NL
    B_fold{ii}=squeeze(Net.B(ii,:,:));
end
opt.B_fold=B_fold;

init_score=valueCalculator(Net,init_state,opt);
    
succ_state=cell(NF,1);
succ_index=2*ones(NF,1);
succ_score=zeros(NF,1);

while(1)
 
    for ii=1:NF
        [succ_state{ii},update]=FindSucc(init_state,scoreRe(ii,:),succ_index(ii),ii);
        
%         succ_state{ii}'
        
        if update==1
            succ_score(ii)=valueCalculator(Net,succ_state{ii},opt);
            
%             succ_score(ii)
            
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

final_state=categorical(init_state);
final_score=init_score;

end

function [succ_state,update]=FindSucc(init_state, score, succ_index,index)

succ_state=init_state;
update=0;
[~,II]=maxk(score, succ_index);

if II(end)>0
    succ_state(index)=II(end);
    update=1;
end

end
