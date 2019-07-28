function initial_point = FindEcForFlow(Net)

[~,I]=sort(Net.prob,2,'descend');
ar = Net.AccessRouter(I(:,1));

list_ec=Construct_EC_List(Net,ar);

initial_point=list_ec(:,1);

end

function list_ec = Construct_EC_List(Net,ar)

NF=length(Net.sk);
NE=length(Net.EdgeCloud);
list_cost=zeros(NF,NE);

for ii = 1:NF
    for jj = 1:NE
        [~,path_cost]=shortestpath(Net.G,ar(ii),Net.EdgeCloud(jj));
        list_cost(ii,jj)=path_cost;
    end
end

[~,list_ec]=sort(list_cost,2);

end