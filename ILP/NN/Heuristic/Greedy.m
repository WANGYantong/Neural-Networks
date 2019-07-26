function solution = Greedy(img)

Net=load('../DataStore/network.mat');
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

solution=FindEcForFlow(Net);

end
