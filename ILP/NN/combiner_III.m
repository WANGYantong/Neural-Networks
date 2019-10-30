function labelUpdate = combiner_III(img,labelOriginal,score)
% call MILP after compression

%% generate one more bound for decision variable x
NF=length(labelOriginal);
NE=length(score)/NF;
scoreRe=reshape(score,[NE,NF])';
scoreRe(scoreRe>0)=1;

%% unzip data for MILP
Net=load(['../DataStore/flow',num2str(NF),'/network.mat']);
[prob,sk,bk,SR,BR]=imageDecoding(img);
Net.prob=prob;
Net.sk=sk;
Net.bk=bk;
Net.SR=SR;
Net.BR=BR;

[NL,NA,~]=size(Net.B);

%% MILP
% if MILP is unsolvable, call combiner_II?

% decision variables
x=optimvar('x',NF,NE,'Type','integer','LowerBound',0,'UpperBound',1);
y=optimvar('y',NF,NL,'Type','integer','LowerBound',0,'UpperBound',1);
z=optimvar('z',NF,NA,NE,'Type','integer','LowerBound',0,'UpperBound',1);

t=optimvar('t',NE,'LowerBound',0);
phi=optimvar('phi',NF,NE,'LowerBound',0);

% constraints 
cache_num_constraint=sum(x,2)<=1;   


end

