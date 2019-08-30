function solution = Greedy(img)

NF=size(img,1);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

list_EC=categorical(FindEcForFlow(Net));
solution=list_EC(:,1);
pointer=ones(NF,1); % position indicator

while(1)
    [spaceValue,label]=spacePenalty(solution,Net);
    
    if spaceValue>0
        pos=find(label==categorical(-1));
        pointer(pos)=pointer(pos)+1;
        solution(pos)=list_EC(pos,pointer(pos));
    else
        break;
    end
    
    if all(pointer==size(list_EC,2))
        break;
    end
    
end

end
