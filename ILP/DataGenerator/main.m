clear
clc

NF=5:5:20;
HARDCORE=20;

[para,data] = Scenario(HARDCORE);

for ii=1:length(NF)
    FlowCase(NF(ii),para,data);
end

