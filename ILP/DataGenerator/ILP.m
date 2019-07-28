function result=ILP(para,data)

NF=length(data.flow);
NL=length(para.graph.Edges.EndNodes);
NA=length(para.AccessRouter);
NE=length(para.EdgeCloud);

%% decision variables
x=optimvar('x',NF,NE,'Type','integer','LowerBound',0,'UpperBound',1);
y=optimvar('y',NF,NL,'Type','integer','LowerBound',0,'UpperBound',1);
z=optimvar('z',NF,NA,NE,'Type','integer','LowerBound',0,'UpperBound',1);

%% constraints
cache_num_constraint=sum(x,2)==1;   

space_size_constraint=data.spaceK*x<=data.spaceR;

path_constraint=sum(z,3)<=1;

x_z=repmat(x,[NA,1,1]);
x_z=reshape(x_z,NF,NA,NE);
z_constraint=z<=x_z;

bandwidth_constraint=data.bandwidthK*y<=data.bandwidthR';

B_y=reshape(data.B,1,NL*NA*NE);
B_y=repmat(B_y,[NF,1]);
B_y=reshape(B_y,NF,NL,NA,NE);
z_y=repmat(z,[NL,1,1,1]);
z_y=reshape(z_y,NF,NL,NA,NE);
y_constraint1=y<=sum(sum(B_y.*z_y,4),3);
y_constraint2=data.M*y>=sum(sum(B_y.*z_y,4),3);

%% objective
objfun1=data.alpha*sum(sum(x));

probability_z=repmat(data.probability,[1,1,NE]);
hopcounter_z=reshape(data.hopcounter,1,NA*NE);
hopcounter_z=repmat(hopcounter_z,[NF,1]);
hopcounter_z=reshape(hopcounter_z,NF,NA,NE);
objfun2=data.beta*sum(sum(sum(probability_z.*hopcounter_z.*z,3),2));

objfun3=data.beta*sum((1-sum(sum(probability_z.*z,3),2))*data.hoptotal);

%% load optimization section
ProCach=optimproblem;

ProCach.Objective=objfun1+objfun2+objfun3;

ProCach.Constraints.Constr1=cache_num_constraint;
ProCach.Constraints.Constr2=space_size_constraint;
ProCach.Constraints.Constr3=path_constraint;
ProCach.Constraints.Constr4=z_constraint;
ProCach.Constraints.Constr5=bandwidth_constraint;
ProCach.Constraints.Constr6=y_constraint1;
ProCach.Constraints.Constr7=y_constraint2;

%% optimal solver
opts=optimoptions('intlinprog','Display','off','MaxTime',3600*2);

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