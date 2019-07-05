function [prob,sk,bk,SR,BR] = imageDecoding(img)

layout=load('../DataStore/layout.mat');

switch layout.image_layout.opts
    
    case 0
        prob=(255-img(layout.image_layout.prob.x,layout.image_layout.prob.y))/255;
        sk=(255-img(layout.image_layout.sprq.x,layout.image_layout.sprq.y))*10/255;
        bk=(255-img(layout.image_layout.bwrq.x,layout.image_layout.bwrq.y))*10/255;
        SR=img(layout.image_layout.space.x,layout.image_layout.space.y)'*50/255;
        BR=img(layout.image_layout.bw.x,layout.image_layout.bw.y)'*100/255;
        
    case {1,2,3}
        prob=(255-img(layout.image_layout.prob.x,layout.image_layout.prob.y))'/255;
        sk=(255-img(layout.image_layout.sprq.x,layout.image_layout.sprq.y))'*10/255;
        bk=(255-img(layout.image_layout.bwrq.x,layout.image_layout.bwrq.y))'*10/255;
        SR=img(layout.image_layout.space.x,layout.image_layout.space.y)'*50/255;
        BR=img(layout.image_layout.bw.x,layout.image_layout.bw.y)'*100/255;
        
    case 4
        prob=(255-img(layout.image_layout.prob.x,layout.image_layout.prob.y))/255;
        sk=(255-img(layout.image_layout.space.x,layout.image_layout.space.y))/255;
        bk=(255-img(layout.image_layout.bandwidth.x,layout.image_layout.bandwidth.y))/255;
%         prob=img(layout.image_layout.prob.x,layout.image_layout.prob.y);
%         sk=img(layout.image_layout.space.x,layout.image_layout.space.y)/10;
%         bk=img(layout.image_layout.bandwidth.x,layout.image_layout.bandwidth.y)/20;
        
        % indicator to represent value normalization case
        SR=0;
        BR=0;
        

end

