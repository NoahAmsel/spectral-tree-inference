% generate a few statistics about the 20 newsgroup datasets
load('/kyb/agbs/harmeling/prj/pcfg/datasets/roweis/20news_w100.mat');

clf

% count for each newsgroup class the number of word appearances
wa = zeros(100, 4);   % matrix counting documents containing particular words
for i = 1:4
  sel = find(newsgroups==i);
  wa(:,i) = sum(documents(:, sel), 2);
  subplot(4,1,i)
  stem(wa(:,i))
end
