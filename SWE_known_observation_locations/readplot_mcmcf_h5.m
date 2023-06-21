%read hd5F files with extensions h5

clc
clear
close all


%% Check if export_fig-master folder is in this current folder,
% otherwise download it
% export_fig function is used to generate high quality plots in pdf or
% other formats
if ~exist('export_fig-master', 'dir')
  url = 'https://github.com/altmany/export_fig/archive/refs/heads/master.zip';
    outfilename = websave([pwd,'/export_fig-master'],url);
    unzip('export_fig-master.zip')
end
addpath([pwd,'/export_fig-master'])
addpath([pwd,'/npy-matlab'])
%%
main_dir = './example';

path_dir = [main_dir, '/mcmc_filtering/restart'];
data_dir = [main_dir,'/data'];

nsimul = 26;
T = 2000; %2160;
dt = 60;
t_freq = 10;
N = 1200;
dgx = 121;
dgy = 121;
dx = 8602.0;
dy = 9258.0;
lat = [17, 27];
lon = [-51, -41];
sig_y = 1.45e-2;

dim2 = dgx * dgy;
dimx = 3*dim2;
Nf = 12;



%%


signal = zeros(dimx,T+1);

filename = [data_dir, '/data.h5'];
for t = 0:T
    signal(:,t+1) = h5read(filename,sprintf('/step_%.8d/signal',t))';
end

%%
filename = [data_dir, '/floater_info.h5'];
floaters_xy(1,:) = h5read(filename,sprintf('/step_%.8d/floaters_xy',0))';
% floaters_height(1,:) = h5read(filename,sprintf('/step_%.8d/floaters_height',0))';
k = 1;
for t = 1:T
    if mod(t,t_freq) == 0
        obs_ind(:,k) = h5read(filename,sprintf('/step_%.8d/obs_indices',t-1))';
        k = k+1;
    end
    floaters_xy(t+1,:) = h5read(filename,sprintf('/step_%.8d/floaters_xy',t))';
%     floaters_height(t+1,:) = h5read(filename,sprintf('/step_%.8d/floaters_height',t))';
end

%%

E_mcmc_h = zeros(dimx, T+1, nsimul);
for h = 0:nsimul-1 
    filename = [path_dir, sprintf('/mcmc_restart_%.8d.h5',h)];
    E_mcmc_h(:,:,h+1) = h5read(filename,'/mcmc_samples')';
end



%%
E_mcmc = mean(E_mcmc_h,3);
%E_mcmc = E_mcmc_h(:,:,1);
%Height
H_signal = zeros(dgx,dgy,T+1);
H_mcmc = zeros(dgx,dgy,T+1);
for t = 1:T+1
    H_signal(:,:,t) = reshape(signal(1:dim2,t),dgx,dgy);
    H_mcmc(:,:,t) = reshape(E_mcmc(1:dim2,t),dgx,dgy);
end

% set(0, 'defaultLegendInterpreter','latex');
% set(0, 'defaultTextInterpreter','latex');
% %set the background of the figure to be white
% set(0,'defaultfigurecolor',[1 1 1])

% figure
% for t=1:T
%     surf(H_mcmc(:,:,t),'CDataMapping','scaled')
%     colormap(parula)
%     title(sprintf('Height - t = %d', t), 'FontSize', 30)
%     axx = gca;
%     zlim([0,2.5])
%     axx.YDir = 'normal';
%     axx.XAxis.FontSize = 20;
%     axx.YAxis.FontSize = 20;
%     
%     drawnow
% end
%%

T1 = T;

min_H = min(min(abs(signal(1:dim2,1:T1))));
max_H = max(max(abs(signal(1:dim2,1:T1))));
min_U = min(min(abs(signal(dim2:2*dim2,1:T1))));
max_U = max(max(abs(signal(dim2:2*dim2,1:T1))));
min_V = min(min(abs(signal(2*dim2:3*dim2,1:T1))));
max_V = max(max(abs(signal(2*dim2:3*dim2,1:T1))));

% errors
abs_err =  abs(signal(:,1:T1)-E_mcmc(:,1:T1));
rel_abs_err = abs_err;
rel_abs_err(1:dim2,:) = abs_err(1:dim2,:)/(max_H-min_H);
rel_abs_err(dim2:2*dim2,:) = abs_err(dim2:2*dim2,:)/(max_U-min_U);
rel_abs_err(2*dim2:end,:) = abs_err(2*dim2:end,:)/(max_V-min_V);

max_rerr = max(rel_abs_err(:));
max_aerr = max(abs_err(:));

%%
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

figure
set(gcf, 'Position',  [100, 100, 1200, 1200])
%h1 = histogram(abs_err(:), 'Normalization', 'probability');
%xlim([0, max_aerr-0.1]);
%xticks(0:0.05:max_aerr-0.1)

h1 = histogram(abs_err(:), 'Normalization', 'probability');
xlim([0, 4*sig_y]);
h1.Normalization = 'probability';
h1.BinWidth = 0.5*sig_y;
a = 0:0.5*sig_y:4*sig_y;
xticks(a)
labels = cell(size(a));
labels{abs(a-sig_y/2)<=1.e-10} = "0.5\sigma_y";
labels{abs(a-sig_y)<=1.e-10} = "\sigma_y";
labels{abs(a-1.5*sig_y)<=1.e-10} = "1.5\sigma_y";
labels{abs(a-2*sig_y)<=1.e-10} = "2\sigma_y";
labels{abs(a-2.5*sig_y)<=1.e-10} = "2.5\sigma_y";
labels{abs(a-3*sig_y)<=1.e-10}= "3\sigma_y";
labels{abs(a-3.5*sig_y)<=1.e-10}= "3.5\sigma_y";
labels{abs(a-4*sig_y)<=1.e-10}= "4\sigma_y";
xticklabels(labels)

ylim([0 1])
yticks(0:0.05:1)


axx = gca;
axx.TickDir = 'out';
axx.XAxis.FontSize = 26;
axx.YAxis.FontSize = 22;
titstr = sprintf('Histogram of Absolute Errors ($\\sigma_y=%.2E$)',sig_y);
title(titstr, 'FontSize', 36)
xlabel('Absolute Error', 'FontSize', 36)
ylabel('Percentage of Occurrence',  'FontSize', 36)


%%


set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])


cmap_ = flipud(getPyPlot_cMap("RdBu_r"));
time = 0:T;

n_subplots = 4;
axis_num = 1;
map = [1 1 1 %white
        0 1 0 %green
       1 0 0 %red
       0 0 0 %blue
       ];
       %1 0 1 %magenta
       %0 0 0 
       %0 0 0
       %0 0 0
       %0 0 0]; %black 
       
map1 = [1 1 1 %white
        0 1 0 %green
       1 0 0 %red
       0 0 1 %blue
       1 0 1 %magenta
       0.5 0.5 0.5
       0 0 0]; %black 

figure
set(gcf, 'Position',  [100, 100, 1600, 1400])

    ax(axis_num) = subplot(n_subplots/2,n_subplots/2,axis_num);
    image(signal(:,1:T1),'CDataMapping','scaled')
    caxis ([min_U,max_U])
    colormap(ax(axis_num),cmap_)
    c1 = colorbar;
    c1.FontSize = 12;
    title('Signal', 'FontSize',25)
    axx = gca;
    axx.YDir = 'normal';
    xlabel('Time', 'FontSize',25)
    ylabel('Coordinates','FontSize',25)
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots/2,n_subplots/2,axis_num);
    image(E_mcmc(:,1:T1),'CDataMapping','scaled')
    caxis ([min_U,max_U])
    colormap(ax(axis_num),cmap_)
    c1 = colorbar;
    c1.FontSize = 12;
    str = sprintf('Filtering using MCMC (Avg. of %d simulations with $N = %d$ samples)',...
                    nsimul,N);
    title(str, 'FontSize', 25)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    xlabel('Time','FontSize',25)
    ylabel('Coordinates','FontSize',25)
    axis_num = axis_num + 1;
    
    ax(axis_num) = subplot(n_subplots/2,n_subplots/2,axis_num);
    image(abs_err,'CDataMapping','scaled')
    caxis ([0,max_aerr])
    colormap(ax(axis_num),map1);%bluewhitered(3))
    c1 = colorbar;
    c1.FontSize = 14; 
    str  = sprintf('Absolute Error');
    title(str, 'FontSize', 25)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;
    axis_num = axis_num + 1;
    
    
    ax(axis_num) = subplot(n_subplots/2,n_subplots/2,axis_num);
    image(rel_abs_err,'CDataMapping','scaled')
    caxis ([0,max_rerr])
    colormap(ax(axis_num),map);%bluewhitered(3))
    c1 = colorbar;
    c1.FontSize = 14; 
    str  = sprintf('Relative Absolute Error');
    title(str, 'FontSize', 25)
    axx = gca;
    axx.YDir = 'normal';
    axx.XAxis.FontSize = 15;
    axx.YAxis.FontSize = 15;




% time = 0:T;
% f1 = figure;
% set(gcf, 'Position',  [100, 100, 1200, 1600])
%     
% for j = 1 : length(obs_ind(:,1))
%     
%     coord = obs_ind(j,1);
%     E_mcmc_coord = squeeze(E_mcmc_h(coord,:,:))';
%     true_mean = mean(signal(coord,:));
%     RRMSE =  sqrt(mean((signal(coord,:) - E_mcmc_coord).^2,1)...
%             /sum(signal(coord,:).^2));
% 
%     [~,time_coord_observed] = find(obs_ind == coord);
%     time_coord_observed = time_coord_observed * t_freq;
% 
% 
%     ylo = -0.15+min(signal(coord,:));
%     yto = 0.15+max(signal(coord,:));
%     subplot(2,1,1)
%     for i = 1:nsimul-1
%         hp = scatter(time,E_mcmc_coord(i,:),4,'filled','MarkerFaceColor','g');
%         hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
%         hold on
%     end
%     h1 = scatter(time,E_mcmc_coord(nsimul,:),4,'filled','MarkerFaceColor','g');
%     hold on
%     h2 = plot(time,signal(coord,:),'k-','LineWidth',3);
%     hold on
%     h3 = plot(time,E_mcmc(coord,:),'r-','LineWidth',3);
%     hold on
%     if ~isempty(time_coord_observed)
%         h4 = xlin(time_coord_observed(1), ylo, ylo + 0.1, 'b',4);
%         for i = 2: length(time_coord_observed)
%             xlin(time_coord_observed(i), ylo, ylo + 0.1, 'b', 4);
%             hold on
%         end
%         legend show
%         legend([h1, h2, h3, h4],{'Simulations','Reference', 'Mean','Time Observed'},...
%                 'Location','northeast', 'color','none')
%         set(legend, 'FontSize', 24, 'Orientation','horizontal')
%     else
%         legend show
%         legend([h1, h2, h3],{'Simulations','Reference', 'Mean'},...
%                 'Location','northeast', 'color','none')
%         set(legend, 'FontSize', 24, 'Orientation','horizontal')
%     end
% 
%     ylim([ylo yto])
%     axx = gca;
%     axx.XAxis.FontSize = 22;
%     axx.YAxis.FontSize = 22;
%     xlabel('$n$', 'FontSize', 36)
%     ylabel(['$E(X_n($',num2str(coord),'$)|data)$'],'FontSize', 36)
%     str = sprintf('Coordinate %d',coord);
%     title(str, 'FontSize', 28)
% 
%     subplot(2,1,2)
%     plot(time, RRMSE, 'b-', 'LineWidth', 3)
%     axx = gca;
%     axx.XAxis.FontSize = 22;
%     axx.YAxis.FontSize = 22;
%     xlabel('$n$', 'FontSize', 36)
%     ylabel('Relative RMSE','FontSize', 36)
%     
%     pause(2)
%     clf(f1)
% end

%plot 2d snapshots of the hight H/U velocity/V velocity
% for i = 1:length(floaters_height(1,:))
%     cc(i,:) = rand(1,3);
% end
%%
cmap_ = flipud(getPyPlot_cMap("RdBu_r"));

set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])


figure
set(gcf, 'Position',  [100, 100, 1000, 1200])
x = linspace(0,dgx*dx,dgx);
y = linspace(0,dgy*dy,dgy);
[X,Y] = meshgrid(x,y);
t0 = 1;

SH0 = reshape(signal(1:dim2,t0),[dgx,dgy]);
SU0 = reshape(signal(dim2+1:2*dim2,t0),[dgx,dgy]);
SV0 = reshape(signal(2*dim2+1:3*dim2,t0),[dgx,dgy]);

H0 = reshape(E_mcmc(1:dim2,t0),[dgx,dgy]);
U0 = reshape(E_mcmc(dim2+1:2*dim2,t0),[dgx,dgy]);
V0 = reshape(E_mcmc(2*dim2+1:3*dim2,t0),[dgx,dgy]);

min_H = min(min(SH0));
max_H = max(max(SH0));
min_U = min(min(SU0));
max_U = max(max(SU0));
min_V = min(min(SV0));
max_V = max(max(SV0));


ax(1) = subplot(3,3,1);
gc = gca();
p1 = image(gc, 'XData',x,'YData',y,'CData',flipud(SU0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S1(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P1(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on;
end
axis tight
title('$U_{Signal}$','FontSize', 36)
colormap(cmap_);
colorbar;
caxis([min_U,max_U]);
set(gca, 'YDir', 'reverse');



ax(2) = subplot(3,3,2);
gc = gca();
p2 = image(gc, 'XData',x,'YData',y,'CData',flipud(SV0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
colormap(cmap_);
title('$V_{Signal}$','FontSize', 36)
colorbar;
caxis([min_V,max_V]);
for fl = 0:Nf-1
    S2(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P2(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
set(gca, 'YDir', 'reverse');

ax(3) = subplot(3,3,3);
gc = gca();
p3 = image(gc, 'XData',x,'YData',y,'CData',flipud(SH0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S3(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P3(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
title('$\eta_{Signal}$', 'FontSize', 36)
colormap(cmap_);
colorbar;
caxis([min_H,max_H]);
set(gca, 'YDir', 'reverse');



ax(4) = subplot(3,3,4);
gc = gca();
p4 = image(gc, 'XData',x,'YData',y,'CData',flipud(U0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S4(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P4(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
colormap(cmap_);
title('$U_{Filter}$','FontSize', 36)
colorbar;
caxis([min_U,max_U]);
set(gca, 'YDir', 'reverse');



ax(5) = subplot(3,3,5);
gc = gca();
p5 = image(gc, 'XData',x,'YData',y,'CData',flipud(V0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S5(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P5(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
colormap(cmap_);
title('$V_{Filter}$','FontSize', 36)
colorbar;
caxis([min_V,max_V]);
set(gca, 'YDir', 'reverse');


ax(6) = subplot(3,3,6);
gc = gca();
p6 = image(gc, 'XData',x,'YData',y,'CData',flipud(H0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S6(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P6(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
colormap(cmap_);
title('$\eta_{Filter}$','FontSize', 36)
colorbar;
caxis([min_H,max_H]);
set(gca, 'YDir', 'reverse');


ax(7) = subplot(3,3,7);
gc = gca();
p7 = image(gc, 'XData',x,'YData',y,'CData',flipud(U0 - SU0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S7(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P7(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
colormap(cmap_);
title('$U_{Filter} - U_{Signal}$','FontSize', 36)
colorbar;
caxis([min_U/2,max_U/2]);
set(gca, 'YDir', 'reverse');



ax(8) = subplot(3,3,8);
gc = gca();
p8 = image(gc, 'XData',x,'YData',y,'CData',flipud(V0 - SV0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S8(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P8(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end
axis tight
colormap(cmap_);
title('$V_{Filter} - V_{Signal}$','FontSize', 36)
colorbar;
caxis([min_V/2,max_V/2]);
set(gca, 'YDir', 'reverse');


ax(9) = subplot(3,3,9);
gc = gca();
p9 = image(gc, 'XData',x,'YData',y,'CData',flipud(H0 - SH0),'CDataMapping','scaled',"Interpolation",'bilinear');
hold on
for fl = 0:Nf-1
    S9(fl+1) = scatter(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),10,"red",'filled');
    P9(fl+1) = plot(floaters_xy(1,2*fl+1),floaters_xy(1,2*fl+2),"--r",'LineWidth',3);
    hold on
end

axis tight
colormap(cmap_);
title('$\eta_{Filter} - \eta_{Signal}$','FontSize', 36)
colorbar;
caxis([-max_H/2,max_H/2]);
set(gca, 'YDir', 'reverse');

tit = sgtitle(sprintf('Snapshots at time = %0.4f hrs',(t0-1)*dt/3600),'FontSize', 36);

for t0 = 1921:1921
    
    SH0 = reshape(signal(1:dim2,t0),[dgx,dgy]);
    SU0 = reshape(signal(dim2+1:2*dim2,t0),[dgx,dgy]);
    SV0 = reshape(signal(2*dim2+1:3*dim2,t0),[dgx,dgy]);

    H0 = reshape(E_mcmc(1:dim2,t0),[dgx,dgy]);
    U0 = reshape(E_mcmc(dim2+1:2*dim2,t0),[dgx,dgy]);
    V0 = reshape(E_mcmc(2*dim2+1:3*dim2,t0),[dgx,dgy]);

    set(p1, 'CData', flipud(SU0));
    set(p2, 'CData', flipud(SV0));
    set(p3, 'CData', flipud(SH0))
    set(p4, 'CData', flipud(U0))
    set(p5, 'CData', flipud(V0))
    set(p6, 'CData', flipud(H0))
    set(p7, 'CData', flipud(U0-SU0))
    set(p8, 'CData', flipud(V0-SV0))
    set(p9, 'CData', flipud(H0-SH0))
    
    for fl = 0:Nf-1
        set(S1(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S2(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S3(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S4(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S5(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S6(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S7(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S8(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
        set(S9(fl+1),'XData',floaters_xy(t0,2*fl+1),...
                        'YData',floaters_xy(t0,2*fl+2))
                    
                    
        set(P1(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P2(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P3(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P4(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P5(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P6(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P7(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P8(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
        set(P9(fl+1),'XData',floaters_xy(1:t0,2*fl+1),...
                        'YData',floaters_xy(1:t0,2*fl+2))
    end
    
    
    simu_time = (t0-1)*dt/3600;
    tit.String = sprintf('Snapshots at time = %0.4f hrs',simu_time);
    pause(0.5)
    drawnow
    if simu_time == 2
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    elseif simu_time == 8
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    elseif simu_time == 14
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    elseif simu_time == 20
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    elseif simu_time == 26
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    elseif simu_time == 32
        export_fig(sprintf('known_loc%d.png', simu_time), '-m2', '-png');
    end
end
    








