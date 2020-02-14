function [cost,ratio] = TCcalculator(img,allocations)

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

allocations=double(string(allocations)); % convert from catgorical to int
[cost,ratio]=valueCalculator(Net,allocations,opt);

end

