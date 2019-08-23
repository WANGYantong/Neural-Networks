function solution = Greedy(img)

NF=size(img,1);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

solution=FindEcForFlow(Net);

end
