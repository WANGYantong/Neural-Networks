function labelUpdate = combiner_II(img,labelOriginal,score)

NF=length(labelOriginal);
NE=length(score)/NF;

scoreRe=reshape(score,[NE,NF])';
[row,col]=find(scoreRe>=0.0001);
value=zeros(size(row));
for ii=1:length(row)
    value(ii)=scoreRe(row(ii),col(ii));
end
[~,ind]=sort(value,'descend');
row=row(ind);
col=categorical(col(ind));

labelUpdate=labelOriginal;
for ii=NF+1:length(row)
    labelUpdate(row(ii))=col(ii);
    valueUpdate=valueCalculator(img,labelUpdate);
    valueOriginal=valueCalculator(img,labelOriginal);
    if valueUpdate<valueOriginal
        labelOriginal=labelUpdate;
    else
        labelUpdate=labelOriginal;
    end
end

end

