function solution = Greedy(img)

NF=size(img,1);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

[opt.NL,opt.NA,opt.NE]=size(Net.B);
opt.mode=0;
opt.NF=NF;
opt.unvalid=-1;
B_fold=cell(opt.NL,1);
for ii=1:opt.NL
    B_fold{ii}=squeeze(Net.B(ii,:,:));
end
opt.B_fold=B_fold;

list_EC=FindEcForFlow(Net);
allocations=list_EC(:,1);
pointer=ones(NF,1); % position indicator

while(1)
    [spaceValue,label]=spacePenalty(allocations,Net,opt);
    
    if spaceValue>0
        pos=find(label==opt.unvalid);
        pointer(pos)=pointer(pos)+1;
        allocations(pos)=list_EC(pos,pointer(pos));
    else
        break;
    end
    
    if all(pointer==size(list_EC,2))
        break;
    end
    
end

solution.allocations=categorical(allocations);
[solution.value,solution.ratio]=valueCalculator(Net,allocations',opt);

end
