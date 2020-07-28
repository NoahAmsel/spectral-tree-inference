%close all

savefig = 0;
methods = {'RG'; 'NJ'; 'CLRG'; 'CLNJ'};
num_methods = length(methods);
line_color = 'gkbr';
line_marker = '+*os';

figure(1); 
for i=1:num_methods
    semilogx(nset/1000,log10(KLdiv(:,i)),[line_color(i), line_marker(i), '-'],'linewidth',2, 'markersize',8)
    hold on;
end
hold off;
xlim([nset(1)/1000, nset(end)/1000]);
legend(methods)
xlabel('Number of Samples (x1000)','FontSize',14)
ylabel('log_{10}(KL Divergence)','FontSize',14)
if(savefig)
    fileName = sprintf('../figures/JMLR/%s%s', graph, '_KLdiv.eps');
    print('-depsc', fileName);
end
    

figure(2); 
for i=1:num_methods
    semilogx(nset/1000,RFdistance(:,i),[line_color(i), line_marker(i), '-'],'linewidth',2, 'markersize',8)
    hold on;
end
hold off;
xlim([nset(1)/1000, nset(end)/1000]);
xlabel('Number of Samples (x1000)','FontSize',14)
ylabel('Robinson-Foulds metric','FontSize',14)
if(savefig)
    fileName = sprintf('../figures/JMLR/%s%s', graph, '_RFdist.eps');
    print('-depsc', fileName);
end

figure(3); 
for i=1:num_methods
    semilogx(nset/1000,1-num_recovery(:,i),[line_color(i), line_marker(i), '-'],'linewidth',2, 'markersize',8)
    hold on;
end
hold off;
xlim([nset(1)/1000, nset(end)/1000]);
ylim([0,1.1]);
%legend(methods)
xlabel('Number of Samples (x1000)','FontSize',14)
ylabel('Rate of Error','FontSize',14)
if(savefig)
    fileName = sprintf('../figures/JMLR/%s%s', graph, '_errorRate.eps');
    print('-depsc', fileName);
end

num_hidden_true = size(adjmat,1)-m;
figure(4); 
for i=1:num_methods
    semilogx(nset/1000,abs(num_hidden(:,i)-num_hidden_true),[line_color(i), line_marker(i), '-'],'linewidth',2, 'markersize',8)
    hold on;
end
%semilogx(nset/1000,(size(adjmat,1)-m)*ones(length(nset),1),'c-.','linewidth',2); 
hold off;
xlim([nset(1)/1000, nset(end)/1000]);
%legend(methods)
%legend([methods; 'correct'])
%legend('RG','NJ','CLRG','CLNJ')
xlabel('Number of Samples (x1000)','FontSize',14)
ylabel('Number of Hidden Nodes','FontSize',14)
if(savefig)
    fileName = sprintf('../figures/JMLR/%s%s', graph, '_numHidden.eps');
    print('-depsc', fileName);
end



%%


close all
%load('./results/3cayley_197_id2.mat')
%load('./results/3cayley_97_id2.mat')
load('./results/regular_diffm_121_id3_10k.mat')
%load('./results/3cayley_94_id1_10k.mat')

figure1 = figure('XVisual',...
    '0x24 (TrueColor, depth 24, RGB mask 0x00ff 0xff00 0xff0000)');
%figure1figure; figure1=plot(1:length(mset),1-num_recovery(:,3),'ks-.','linewidth',2,'markersize',8)
axes1 = axes('Parent',figure1,...
    'XTickLabel',{'2 (82)','3 (109)','4 (118)','5 (121)'},...
    'XTick',[1 2 3 4],...
    'XMinorTick','off',...
    'FontSize',14,...
    'FontName','Arial');
hold on;
box('on');
plot(1:length(mset),1-num_recovery(:,3),'Parent',axes1,'MarkerSize',8,...
    'LineWidth',2);
xlabel('Number of Observed Scales (Nodes)','FontSize',16,'FontName','Arial')
ylabel('Error Rate','FontSize',16,'FontName','Arial')
n1 = n;

%load('./results/3cayley_197_id2_50k.mat')
%load('./results/3cayley_97_id4.mat')
load('./results/regular_diffm_121_id3_20k.mat')
%load('./results/3cayley_94_id1_50k.mat')
hold on
plot(1:length(mset),1-num_recovery(:,3),'b>:','linewidth',2,'markersize',10)
legend([num2str(n1/1000) 'k'],[num2str(n/1000) 'k'])
hold off
n2 = n;

load('./results/regular_diffm_121_id3_50k.mat')
%load('./results/3cayley_94_id1_50k.mat')
hold on
plot(1:length(mset),1-num_recovery(:,3),'ro-','linewidth',2,'markersize',10)
legend([num2str(n1/1000) 'k'],[num2str(n2/1000) 'k'],[num2str(n/1000) 'k'])
hold off

% hold on
% load('./results/3cayley_97_id4.mat')
% plot(1:length(mset),1-num_recovery,'o-','linewidth',2)
% legend('Recursive','CL+Blind','CL+Recursive')
% xlabel('Number of Samples (x1000)','FontSize',14)
% ylabel('Rate of Error','FontSize',14)
