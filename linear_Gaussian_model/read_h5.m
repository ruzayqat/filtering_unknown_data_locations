%read hd5F files with extensions h5

clc
clear
close all

main_KF = './example_KF';
main_mcmc = "./example_mcmc";
main_enkf = "./example_enkf";
main_etkf = "./example_etkf";
main_estkf = "./example_estkf";

data_dir = strcat(main_KF, '/data');
kf_dir = strcat(main_KF, '/kf');
enkf_dir = strcat(main_enkf, '/enkf');
etkf_dir = strcat(main_etkf, '/etkf');
estkf_dir = strcat(main_estkf, '/estkf');
mcmc_dir = strcat(main_mcmc,'/mcmc_filtering');


nsimul = 26;
T = 100;
N = 350; %for mcmc method
coord = 3;
dimx = 16000;

sig_x = 0.05;
sig_y = 0.05;

mcmc_h = zeros(dimx,T+1, nsimul);
for h = 0:nsimul -1 
    filename = strcat(mcmc_dir, sprintf('/mcmc_%.8d.h5',h));
    mcmc_h(:,:,h+1) = h5read(filename,'/mcmc_filter')';
end

filename = strcat(kf_dir, '/kalman_filter.h5');
KF = h5read(filename,'/kalman_filter')';
filename = strcat(etkf_dir,'/etkf.h5');
ETKF = h5read(filename,'/etkf')';
filename = strcat(estkf_dir, '/estkf.h5');
ESTKF = h5read(filename,'/estkf')';
filename = strcat(enkf_dir,'/enkf.h5');
EnKF = h5read(filename,'/enkf')';
    
mcmcf = mean(mcmc_h,3);
mcmcf1 = squeeze(mcmc_h(coord,:,:));


%% errors
min_KF = min(abs(KF),[],'all');
max_KF = max(abs(KF),[],'all');
L2_KF = norm(KF);
L2_KF1 = norm(KF(coord,:));


abs_err_mcmc = abs(mcmcf - KF);
max_abs_err_mcmc = max(abs_err_mcmc, [], 'all');
rel_err_mcmc = abs_err_mcmc./(max_KF - min_KF);
max_rel_err_mcmc = max(rel_err_mcmc,[],'all');
L2_err_mcmc = norm(mcmcf - KF)/L2_KF;
L2_err_mcmc1 = norm(rel_err_mcmc(coord,:))/L2_KF1;
mse_mcmc = mean((mcmc_h(coord,T+1,:) - KF(coord, T+1)).^2);

abs_err_enkf = abs(EnKF - KF);
max_abs_err_enkf = max(abs_err_enkf, [], 'all');
rel_err_enkf = abs_err_enkf./(max_KF - min_KF);
max_rel_err_enkf = max(rel_err_enkf,[],'all');
L2_err_enkf = norm(EnKF - KF)/L2_KF;
L2_err_enkf1 = norm(rel_err_enkf(coord,:))/L2_KF1;
mse_enkf = mean((EnKF(coord,T+1) - KF(coord, T+1)).^2);


abs_err_etkf = abs(ETKF - KF);
max_abs_err_etkf = max(abs_err_etkf, [], 'all');
rel_err_etkf = abs_err_etkf./(max_KF - min_KF);
max_rel_err_etkf = max(rel_err_etkf,[],'all');
L2_err_etkf = norm(ETKF - KF)/L2_KF;
L2_err_etkf1 = norm(rel_err_etkf(coord,:))/L2_KF1;
mse_etkf = mean((ETKF(coord,T+1) - KF(coord, T+1)).^2);

abs_err_estkf = abs(ESTKF - KF);
max_abs_err_estkf = max(abs_err_estkf, [], 'all');
rel_err_estkf = abs_err_estkf./(max_KF - min_KF);
max_rel_err_estkf = max(rel_err_estkf,[],'all');
L2_err_estkf = norm(ESTKF - KF)/L2_KF;
L2_err_estkf1 = norm(rel_err_estkf(coord,:))/L2_KF1;
mse_estkf = mean((ESTKF(coord,T+1) - KF(coord, T+1)).^2);

fprintf('mse_mcmc = %.3E, mse_enkf = %.3E, mse_etkf = %.3E, mse_sretkf = %.3E\n',...
    mse_mcmc,mse_enkf,mse_etkf,mse_estkf)

% fprintf('mse_mcmc = %.3E, mse_enkf = %.3E, mse_sretkf = %.3E\n',...
%     mse_mcmc,mse_enkf,mse_sretkf)
%%
set(0, 'defaultLegendInterpreter','latex');
set(0, 'defaultTextInterpreter','latex');
%set the background of the figure to be white
set(0,'defaultfigurecolor',[1 1 1])


%%
figure
titstr = {'MCMC-Filter: Absolute Error','MCMC-Filter: Relative Error'};
set(gcf, 'Position',  [100, 100, 900, 400])
    
for i = 1 : 2
    ax(i) = subplot(1,2,i);
    if i == 1
        h1 = histogram(abs_err_mcmc);
    elseif i == 2
        h1 = histogram(rel_err_mcmc);
    end
    
    xlim([0, 3*sig_x]);
    a = 0:sig_x/2:3*sig_x;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_x/2)<=1.e-10} = "0.5\sigma_x";
    labels{abs(a-sig_x)<=1.e-10} = "\sigma_x";
    labels{abs(a-1.5*sig_x)<=1.e-10} = "1.5\sigma_x";
    labels{abs(a-2*sig_x)<=1.e-10} = "2\sigma_x";
    labels{abs(a-2.5*sig_x)<=1.e-10} = "2.5\sigma_x";
    labels{abs(a-3*sig_x)<=1.e-10} = "3\sigma_x";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_x/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 18;
    axx.YAxis.FontSize = 18;
    title(titstr{i}, 'FontSize', 34)
    
end


figure
titstr = {'EnKF: Absolute Error','EnKF: Relative Error'};
set(gcf, 'Position',  [100, 600, 900, 400])
    
for i = 1 : 2
    ax(i) = subplot(1,2,i);
    if i == 1
        h1 = histogram(abs_err_enkf);
    elseif i == 2
        h1 = histogram(rel_err_enkf);
    end
    
    xlim([0, 3*sig_x]);
    a = 0:sig_x/2:3*sig_x;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_x/2)<=1.e-10} = "0.5\sigma_x";
    labels{abs(a-sig_x)<=1.e-10} = "\sigma_x";
    labels{abs(a-1.5*sig_x)<=1.e-10} = "1.5\sigma_x";
    labels{abs(a-2*sig_x)<=1.e-10} = "2\sigma_x";
    labels{abs(a-2.5*sig_x)<=1.e-10} = "2.5\sigma_x";
    labels{abs(a-3*sig_x)<=1.e-10} = "3\sigma_x";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_x/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 18;
    axx.YAxis.FontSize = 18;
    title(titstr{i}, 'FontSize', 34)
    
  
end



figure
titstr = {'ESTKF: Absolute Error','ESTKF: Relative Error'};
set(gcf, 'Position',  [1000, 700, 900, 400])
    
for i = 1 : 2
    ax(i) = subplot(1,2,i);
    if i == 1
        h1 = histogram(abs_err_estkf);
    elseif i == 2
        h1 = histogram(rel_err_estkf);
    end
    
    xlim([0, 3*sig_x]);
    a = 0:sig_x/2:3*sig_x;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_x/2)<=1.e-10} = "0.5\sigma_x";
    labels{abs(a-sig_x)<=1.e-10} = "\sigma_x";
    labels{abs(a-1.5*sig_x)<=1.e-10} = "1.5\sigma_x";
    labels{abs(a-2*sig_x)<=1.e-10} = "2\sigma_x";
    labels{abs(a-2.5*sig_x)<=1.e-10} = "2.5\sigma_x";
    labels{abs(a-3*sig_x)<=1.e-10} = "3\sigma_x";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_x/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 18;
    axx.YAxis.FontSize = 18;
    title(titstr{i}, 'FontSize', 34)
    
end





figure
titstr = {'ETKF: Absolute Error','ETKF: Relative Error'};
set(gcf, 'Position',  [1000, 100, 900, 400])
    
for i = 1 : 2
    ax(i) = subplot(1,2,i);
    if i == 1
        h1 = histogram(abs_err_etkf);
    elseif i == 2
        h1 = histogram(rel_err_etkf);
    end
    
    xlim([0, 3*sig_x]);
    a = 0:sig_x/2:3*sig_x;
    xticks(a)
    labels = cell(size(a));
    labels{abs(a-sig_x/2)<=1.e-10} = "0.5\sigma_x";
    labels{abs(a-sig_x)<=1.e-10} = "\sigma_x";
    labels{abs(a-1.5*sig_x)<=1.e-10} = "1.5\sigma_x";
    labels{abs(a-2*sig_x)<=1.e-10} = "2\sigma_x";
    labels{abs(a-2.5*sig_x)<=1.e-10} = "2.5\sigma_x";
    labels{abs(a-3*sig_x)<=1.e-10} = "3\sigma_x";
    xticklabels(labels)
    
    ylim([0 1])
    yticks(0:0.05:1)
    h1.Normalization = 'probability';
    h1.BinWidth = sig_x/2;
    
    axx = gca;
    axx.TickDir = 'out';
    axx.XAxis.FontSize = 18;
    axx.YAxis.FontSize = 18;
    title(titstr{i}, 'FontSize', 34)
    
end



% %% Plot for coord
% time=0:T;
% 
% figure
% set(gcf, 'Position',  [100, 100, 1200, 1600])
% 
% for i = 1:nsimul-1
%     hp = scatter(time,mcmcf1(:,i),9,'filled','MarkerFaceColor','g');
%     hp.Annotation.LegendInformation.IconDisplayStyle = 'off';
%     hold on
% end
% scatter(time,mcmcf1(:,nsimul),9,'filled','MarkerFaceColor','g', 'DisplayName', 'MCMC-filter Simuls.');
% hold on
% plot(time,KF(coord,:),'k-','DisplayName', 'Kalman Filter','LineWidth',3)
% hold on
% plot(time,ETKF(coord,:),'c-','DisplayName', 'ETKF Filter','LineWidth',2)
% hold on
% plot(time,ESTKF(coord,:),'m-','DisplayName', 'ESTKF Filter','LineWidth',2)
% hold on
% plot(time,EnKF(coord,:),'b-','DisplayName', 'Avg EnKF','LineWidth',2)
% hold on
% plot(time,mcmcf(coord,:),'r-','DisplayName', 'Avg MCMC-Filter','LineWidth',2)
% hold off
% str1 = sprintf('Expectations of $\\varphi(x_n)=x_n^%d$ w.r.t. the MCMC-filter distribution',coord);
% str2 = sprintf('over %d simulations with %d particles ($L_2$-error $=%.2E$)',nsimul, N, L2_err_mcmc1);
% title({str1,str2},'FontSize', 30)
% legend show
% legend('Location','northeast', 'color','none')
% set(legend, 'FontSize', 18, 'Orientation','horizontal')
% ylim([-0.2+min(KF(coord,:)) 0.2+max(KF(coord,:))])
% axx = gca;
% axx.XAxis.FontSize = 18;
% axx.YAxis.FontSize = 17;
% xlabel('$n$', 'FontSize', 30)
% ylabel('$E(X_n^1\,|data)$','FontSize', 30)
% 
