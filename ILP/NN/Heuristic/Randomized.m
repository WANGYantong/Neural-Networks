function solution = Randomized(img,original)

rng(1);

NF=size(img,1);

Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

NE=length(Net.EdgeCloud);

TIMES_HARDCODE=200;
original=categorical(original);
update=original;

opt.mode=0;

for ii=1:TIMES_HARDCODE
    for jj=1:NF
        if rand>=0.5
            update(jj)=categorical(randi(NE));
        end
    end
    valueUpdate=valueCalculator(img,update,opt);
    valueOriginal=valueCalculator(img,original,opt);
    if valueUpdate<valueOriginal
        original=update;
    else
        update=original;
    end
end

solution=update;

end

