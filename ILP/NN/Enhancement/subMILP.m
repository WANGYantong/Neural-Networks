function result = subMILP(img,labelOriginal,score,Net)
% call MILP after compression

%% generate one more bound for decision variable x
NF=length(labelOriginal);
NE=length(score)/NF;
scoreRe=reshape(score,[NE,NF])';
scoreRe(scoreRe>=1e-3)=1;
scoreRe=round(scoreRe);

%% unzip data for MILP
[Net.prob,Net.sk,Net.bk,Net.SR,Net.BR]=imageDecoding(img);
Net.M=1e6;

[NL,NA,~]=size(Net.B);

base_x=sum(scoreRe==0,'all');
base_t=sum(sum(scoreRe==1,1)==0);
result.num_var=base_x*2+base_x*NA+base_t;
%% MILP
% if MILP is unsolvable, call combiner_II?

% decision variables
x=optimvar('x',NF,NE,'Type','integer','LowerBound',0,'UpperBound',1);
y=optimvar('y',NF,NL,'Type','integer','LowerBound',0,'UpperBound',1);
z=optimvar('z',NF,NA,NE,'Type','integer','LowerBound',0,'UpperBound',1);

t=optimvar('t',NE,'LowerBound',0);
phi=optimvar('phi',NF,NE,'LowerBound',0);

% constraints 
cache_num_constraint=sum(x,2)==1;   

if any(Net.SR) || any(Net.BR)
    s_x=repmat(Net.sk,[1,NE]);
    space_size_constraint=sum(s_x.*x,1)<=Net.SR';
    
    b_y=repmat(Net.bk,[1,NL]);
    bandwidth_constraint=sum(b_y.*y,1)<=Net.BR';
    
    space_uti=transpose(transpose(1./Net.SR)*Net.sk);
    t_constraint=t'-sum(space_uti.*phi,1)==1;
else
    space_size_constraint=sum(Net.sk.*x,1)<=1;
    bandwidth_constraint=sum(Net.bk.*y,1)<=1;
    t_constraint=t'-sum(Net.sk.*phi,1)==1;
end

path_constraint=sum(z,3)<=1;

x_z=repmat(x,[NA,1,1]);
x_z=reshape(x_z,NF,NA,NE);
z_constraint1=z<=x_z;

probability_z=repmat(Net.prob,[1,1,NE]);
z_constraint2=z<=Net.M*probability_z;

B_y=reshape(Net.B,1,NL*NA*NE);
B_y=repmat(B_y,[NF,1]);
B_y=reshape(B_y,NF,NL,NA,NE);
z_y=repmat(z,[NL,1,1,1]);
z_y=reshape(z_y,NF,NL,NA,NE);
y_constraint1=y<=sum(sum(B_y.*z_y,4),3);
y_constraint2=Net.M*y>=sum(sum(B_y.*z_y,4),3);

t_phi=repmat(t',[NF,1]);
phi_constraint1=phi<=t_phi;
phi_constraint2=phi<=Net.M*x;
phi_constraint3=phi>=Net.M*(x-1)+t_phi;

% the new constraint added according to the output of CNN
CNN_constraint=x<=scoreRe;

%% objective
objfun1=Net.alpha*sum(sum(phi));

hopcounter_z=reshape(Net.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);
objfun2=Net.beta*sum(sum(sum(probability_z.*hopcounter_z.*z,3),2));

objfun3=Net.gamma*sum((1-sum(sum(probability_z.*z,3),2))*Net.hoptotal);

%% load optimization section
ProCach=optimproblem;

ProCach.Objective=objfun1+objfun2+objfun3;

ProCach.Constraints.Constr1=cache_num_constraint;
ProCach.Constraints.Constr2=space_size_constraint;
ProCach.Constraints.Constr3=path_constraint;
ProCach.Constraints.Constr4=z_constraint1;
ProCach.Constraints.Constr5=z_constraint2;
ProCach.Constraints.Constr6=bandwidth_constraint;
ProCach.Constraints.Constr7=y_constraint1;
ProCach.Constraints.Constr8=y_constraint2;
ProCach.Constraints.Constr9=phi_constraint1;
ProCach.Constraints.Constr10=phi_constraint2;
ProCach.Constraints.Constr11=phi_constraint3;
ProCach.Constraints.Constr12=t_constraint;

ProCach.Constraints.Constr13=CNN_constraint;

%% optimal solver
opts=optimoptions('intlinprog','Display','off','Heuristics','none','MaxTime',3600*12);

tic;
[sol,fval,exitflag,output]=solve(ProCach,'Options',opts);
running_time=toc;

if isempty(sol)
    disp('The solver did not return a solution.')
    return
end

result.time=running_time;
result.sol=sol;
result.fval=fval;
result.exitflag=exitflag;
result.output=output;

[s1,t1]=find(round(sol.x));

[~,II]=sort(s1);
result.allocations=categorical(t1(II));

end

