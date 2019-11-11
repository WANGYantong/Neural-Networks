function result=MILP(para,data)

NF=length(data.flow);
NL=length(para.graph.Edges.EndNodes);
NA=length(para.AccessRouter);
NE=length(para.EdgeCloud);

%% decision variables
x=optimvar('x',NF,NE,'Type','integer','LowerBound',0,'UpperBound',1);
y=optimvar('y',NF,NL,'Type','integer','LowerBound',0,'UpperBound',1);
z=optimvar('z',NF,NA,NE,'Type','integer','LowerBound',0,'UpperBound',1);

t=optimvar('t',NE,'LowerBound',0);
phi=optimvar('phi',NF,NE,'LowerBound',0);

%% constraints
cache_num_constraint=sum(x,2)<=1;   

space_size_constraint=data.spaceK*x<=data.spaceR;

path_constraint=sum(z,3)<=1;

x_z=repmat(x,[NA,1,1]);
x_z=reshape(x_z,NF,NA,NE);
z_constraint1=z<=x_z;

probability_z=repmat(data.probability,[1,1,NE]);
z_constraint2=z<=data.M*probability_z;

bandwidth_constraint=data.bandwidthK*y<=data.bandwidthR';

B_y=reshape(data.B,1,NL*NA*NE);
B_y=repmat(B_y,[NF,1]);
B_y=reshape(B_y,NF,NL,NA,NE);
z_y=repmat(z,[NL,1,1,1]);
z_y=reshape(z_y,NF,NL,NA,NE);
y_constraint1=y<=sum(sum(B_y.*z_y,4),3);
y_constraint2=data.M*y>=sum(sum(B_y.*z_y,4),3);

t_phi=repmat(t',[NF,1]);
phi_constraint1=phi<=t_phi;
phi_constraint2=phi<=data.M*x;
phi_constraint3=phi>=data.M*(x-1)+t_phi;

space_uti=transpose(transpose(1./data.spaceR)*data.spaceK);
t_constraint=t'-sum(space_uti.*phi,1)==1;

%% objective
objfun1=data.alpha*sum(sum(phi));

hopcounter_z=reshape(data.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);
objfun2=data.beta*sum(sum(sum(probability_z.*hopcounter_z.*z,3),2));

objfun3=data.gamma*sum((1-sum(sum(probability_z.*z,3),2))*data.hoptotal);

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

%% optimal solver
opts=optimoptions('intlinprog','Display','off','Heuristics','advanced','MaxTime',3600*12);

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
result.allocations=t1(II);

end