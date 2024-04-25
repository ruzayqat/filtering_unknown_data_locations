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



%%


nsimul = 26;
T = 100;
N = 1000; %for mcmc method
coord = 4;
dimx = 16000; %20736;%16384; %1600;
dimy = 16000; %5184; %4096; %400;

sig_x = 5e-2;
sig_y = 5e-2;


main_folder = sprintf('./example_dx=%d_dy=%d_sigx=%.2E_sigy=%.2E',dimx, dimy, sig_x, sig_y);

data_dir = strcat(main_folder, '/data');
kf_dir = strcat(main_folder, '/kf');
smcmc_dir = strcat(main_folder,'/smcmc');
enkf_dir = strcat(main_folder, '/enkf');
etkf_dir = strcat(main_folder, '/etkf');
estkf_dir = strcat(main_folder, '/estkf');
enkf_loc_dir = strcat(main_folder, '/enkf_loc');




filename = strcat(kf_dir, '/kalman_filter.h5');
KF = h5read(filename,'/kalman_filter')';

smcmc_h = zeros(dimx,T+1, nsimul);
for h = 0:nsimul -1 
    filename = strcat(smcmc_dir, sprintf('/smcmc_%.8d.h5',h));
    smcmc_h(:,:,h+1) = h5read(filename,'/smcmc_filter')';
end
smcmcf = mean(smcmc_h,3);
smcmcf1 = squeeze(smcmc_h(coord,:,:)); %all simulations at one coordinate


filename = strcat(enkf_dir,'/enkf.h5');
EnKF = h5read(filename,'/enkf')';

filename = strcat(etkf_dir,'/etkf.h5');
ETKF = h5read(filename,'/etkf')';

filename = strcat(estkf_dir, '/estkf.h5');
ESTKF = h5read(filename,'/estkf')';   

filename = strcat(enkf_dir,'/enkf.h5');
EnKF_Loc = h5read(filename,'/enkf_loc')';

%% errors
min_KF = min(abs(KF),[],'all');
max_KF = max(abs(KF),[],'all');
L2_KF = norm(KF);
L2_KF1 = norm(KF(coord,:));


abs_err_smcmc = abs(smcmcf - KF);
max_abs_err_smcmc = max(abs_err_smcmc, [], 'all');
L2_err_smcmc = norm(smcmcf - KF)/L2_KF;


abs_err_enkf = abs(EnKF - KF);
max_abs_err_enkf = max(abs_err_enkf, [], 'all');
L2_err_enkf = norm(EnKF - KF)/L2_KF;


abs_err_etkf = abs(ETKF - KF);
max_abs_err_etkf = max(abs_err_etkf, [], 'all');
L2_err_etkf = norm(ETKF - KF)/L2_KF;


abs_err_estkf = abs(ESTKF - KF);
max_abs_err_estkf = max(abs_err_estkf, [], 'all');
L2_err_estkf = norm(ESTKF - KF)/L2_KF;


abs_err_enkf_loc = abs(EnKF_Loc - KF);
max_abs_err_enkf_loc = max(abs_err_enkf_loc, [], 'all');
L2_err_enkf_loc = norm(EnKF_Loc - KF)/L2_KF;

%%
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])

figure
set(gcf, 'Position',  [100, 600, 800, 600])
 
h1 = histogram(abs_err_smcmc(:), 'Normalization', 'probability');
xlim([0, 4*sig_y]);
h1.Normalization = 'probability';
h1.BinWidth = 0.5*sig_y;
a = 0:0.5*sig_y:4*sig_y;
xticks(a+0.5)
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
titstr = sprintf('SMCMC: Histogram of Absolute Errors ($\\sigma_y=%.2E$)',sig_y);
title(titstr, 'FontSize', 36)
xlabel('Absolute Error', 'FontSize', 36)
ylabel('Percentage of Occurrence',  'FontSize', 36)
    
export_fig(sprintf('Hist_SMCMC_d=%d_dy=%d.png', dimx,dimy), '-m3', '-png');


%%   

    
    figure
    set(gcf, 'Position',  [100, 600, 800, 600])
    h1 = histogram(abs_err_enkf);
    
    
    xlim([0, 3*sig_y]);
    a = 0:sig_y/2:3*sig_y;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_y/2)<=1.e-10} = "0.5\sigma_y";
    labels{abs(a-sig_y)<=1.e-10} = "\sigma_y";
    labels{abs(a-1.5*sig_y)<=1.e-10} = "1.5\sigma_y";
    labels{abs(a-2*sig_y)<=1.e-10} = "2\sigma_y";
    labels{abs(a-2.5*sig_y)<=1.e-10} = "2.5\sigma_y";
    labels{abs(a-3*sig_y)<=1.e-10} = "3\sigma_y";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_y/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 28;
    axx.YAxis.FontSize = 22;
    title(sprintf('EnKF: Histogram of Absolute Errors ($\\bar{\\sigma}_y=%.2E$)',sig_y), 'FontSize', 27)
    xlabel('Absolute Error', 'FontSize', 30)
    ylabel('Percentage of Occurrence',  'FontSize', 30)
    
    export_fig(sprintf('Hist_EnKF_d=%d_dy=%d.png', dimx,dimy), '-m3', '-png');



    
    
    figure
    set(gcf, 'Position',  [100, 600, 800, 600])
    h1 = histogram(abs_err_estkf);
    
    
    xlim([0, 3*sig_x]);
    a = 0:sig_y/2:3*sig_y;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_y/2)<=1.e-10} = "0.5\sigma_y";
    labels{abs(a-sig_y)<=1.e-10} = "\sigma_y";
    labels{abs(a-1.5*sig_y)<=1.e-10} = "1.5\sigma_y";
    labels{abs(a-2*sig_y)<=1.e-10} = "2\sigma_y";
    labels{abs(a-2.5*sig_y)<=1.e-10} = "2.5\sigma_y";
    labels{abs(a-3*sig_y)<=1.e-10} = "3\sigma_y";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_y/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 28;
    axx.YAxis.FontSize = 22;
    title(sprintf('ESTKF: Histogram of Absolute Errors ($\\bar{\\sigma}_y=%.2E$)',sig_y), 'FontSize', 27)
    xlabel('Absolute Error', 'FontSize', 30)
    ylabel('Percentage of Occurrence',  'FontSize', 30)
    
    export_fig(sprintf('Hist_ESTKF_d=%d_dy=%d.png', dimx,dimy), '-m3', '-png');


    
    
    
    figure
    set(gcf, 'Position',  [100, 600, 800, 600])
    h1 = histogram(abs_err_etkf);
    
    
    xlim([0, 3*sig_y]);
    a = 0:sig_y/2:3*sig_y;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_y/2)<=1.e-10} = "0.5\sigma_y";
    labels{abs(a-sig_y)<=1.e-10} = "\sigma_y";
    labels{abs(a-1.5*sig_y)<=1.e-10} = "1.5\sigma_y";
    labels{abs(a-2*sig_y)<=1.e-10} = "2\sigma_y";
    labels{abs(a-2.5*sig_y)<=1.e-10} = "2.5\sigma_y";
    labels{abs(a-3*sig_y)<=1.e-10} = "3\sigma_y";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_y/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 28;
    axx.YAxis.FontSize = 22;
    title(sprintf('ETKF: Histogram of Absolute Errors ($\\bar{\\sigma}_y=%.2E$)',sig_y)...
        , 'FontSize', 27)
    xlabel('Absolute Error', 'FontSize', 30)
    ylabel('Percentage of Occurrence',  'FontSize', 30)
    
    export_fig(sprintf('Hist_ETKF_d=%d_dy=%d.png', dimx,dimy), '-m3', '-png');
   

    
    figure
    set(gcf, 'Position',  [100, 600, 800, 600])
    h1 = histogram(abs_err_enkf_loc);
    
    
    xlim([0, 3*sig_y]);
    a = 0:sig_y/2:3*sig_y;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_y/2)<=1.e-10} = "0.5\sigma_y";
    labels{abs(a-sig_y)<=1.e-10} = "\sigma_y";
    labels{abs(a-1.5*sig_y)<=1.e-10} = "1.5\sigma_y";
    labels{abs(a-2*sig_y)<=1.e-10} = "2\sigma_y";
    labels{abs(a-2.5*sig_y)<=1.e-10} = "2.5\sigma_y";
    labels{abs(a-3*sig_y)<=1.e-10} = "3\sigma_y";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_y/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 28;
    axx.YAxis.FontSize = 22;
    title(sprintf('EnKF Local: Histogram of Absolute Errors ($\\bar{\\sigma}_y=%.2E$)',sig_y)...
        , 'FontSize', 27)
    xlabel('Absolute Error', 'FontSize', 30)
    ylabel('Percentage of Occurrence',  'FontSize', 30)
    
    export_fig(sprintf('Hist_EnKF_Loc_d=%d_dy=%d.png', dimx,dimy), '-m3', '-png');


%% Plot for coord all filters
time= 20:50;%0:T;
range = time+1;
DG = [0 0.5 0];

figure
set(gcf, 'Position',  [100, 100, 1200, 1600])

% for i = 1:nsimul-1
%     hp = scatter(time,smcmcf1(:,i),9,'filled','MarkerFaceColor','g');
%     hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
%     hold on
% end
% scatter(time,mcmcf1(:,nsimul),9,'filled','MarkerFaceColor','g', 'DisplayName', 'MCMC-filter Simuls.');
% hold on
plot(time,KF(coord,range),'k-','DisplayName', 'KF','LineWidth',4)
hold on
plot(time,smcmcf(coord,range),'r-','DisplayName', 'SMCMC','LineWidth',4)
hold on
plot(time,EnKF_Loc(coord,range),'Color',DG,'DisplayName', 'LEnKF','LineWidth',4)
hold on
plot(time,EnKF(coord,range),'b--','DisplayName', 'EnKF','LineWidth',3)
hold on
plot(time,ETKF(coord,range),'c--','DisplayName', 'ETKF','LineWidth',3)
hold on
plot(time,ESTKF(coord,range),'m--','DisplayName', 'ESTKF','LineWidth',3)
hold off
str1 = sprintf('Expectations of $\\varphi(z_n)=z_n^{%d}$ w.r.t. the different filtering distributions',coord);
title(str1,'FontSize', 46)
legend show
legend('Location','northeast', 'color','none')
set(legend, 'FontSize', 42, 'Orientation','horizontal', 'Location', 'NorthWest')
ylim([-0.4+min(KF(coord,range)) 0.5+max(KF(coord,range))]) %[0.9290 0.6940 0.1250]
axx = gca;
axx.XAxis.FontSize = 30;
axx.YAxis.FontSize = 30;
xlabel('$n$', 'FontSize', 46)
yyy= sprintf('$E(Z_n^{%d}|data)$',coord);
ylabel(yyy,'FontSize', 46)

export_fig(sprintf('Coord=%d_d=%d_dy=%d_all.png', coord, dimx,dimy), '-m3', '-png');


%% Plot for coord
time= 30:60;%0:T;
range = time+1;
DG = [0 0.5 0];

figure
set(gcf, 'Position',  [100, 100, 1200, 1600])

plot(time,KF(coord,range),'k-','DisplayName', 'KF','LineWidth',4)
hold on
plot(time,EnKF_Loc(coord,range),'Color',DG,'DisplayName', 'LEnKF','LineWidth',4)
hold on
plot(time,smcmcf(coord,range),'r-','DisplayName', 'SMCMC','LineWidth',4)
hold off

str1 = sprintf('Expectations of $\\varphi(z_n)=z_n^{%d}$ w.r.t. LEnKF and SMCMC filters',coord);
title(str1,'FontSize', 46)
legend show
legend('Location','northeast', 'color','none')
set(legend, 'FontSize', 42, 'Orientation','horizontal', 'Location', 'NorthWest')
ylim([-0.1+min(KF(coord,range)) 0.1+max(KF(coord,range))]) %[0.9290 0.6940 0.1250]
axx = gca;
axx.XAxis.FontSize = 30;
axx.YAxis.FontSize = 30;
xlabel('$n$', 'FontSize', 46)
yyy= sprintf('$E(Z_n^{%d}|data)$',coord);
ylabel(yyy,'FontSize', 46)

export_fig(sprintf('Coord=%d_d=%d_dy=%d_LEnKF_vs_SMCMC.png', coord, dimx,dimy), '-m3', '-png');
