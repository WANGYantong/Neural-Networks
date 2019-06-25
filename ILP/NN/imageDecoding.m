function [prob,sk,bk,SR,BR] = imageDecoding(img)

layout=load('../DataStore/layout.mat');

switch layout.opts
    
    case {0,1,2,3}
        prob=img(255-img());

end

